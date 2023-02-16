import config
import polars as pl
from datetime import timedelta


def get_data(
    transaction_path: str = config.transactions_path,
    relation_path: str = config.relations_path,
) -> pl.DataFrame:
    """
    Load the data.

    - Args:
    -------
    - transaction_path: path for tran.
    - relation_path: path for client relation data.

    Returns:
    --------
    - Dataframe of transactions and relationship data.
    """
    transactions = (
        pl.read_parquet(transaction_path)
        .lazy()
        .with_columns(
            [
                pl.col(["date_order", "date_invoice"]).str.strptime(
                    pl.Date, fmt="%Y-%m-%d"
                ),
                pl.col("order_channel").cast(pl.Categorical()),
            ]
        )
    )

    relationsships = (
        pl.read_csv(relation_path)
        .lazy()
        .with_columns(
            pl.col("quali_relation").cast(pl.Categorical).to_physical().cast(pl.UInt8)
        )
    )

    return (
        transactions.join(
            other=relationsships, left_on="client_id", right_on="client_id"
        )
    ).collect()


def get_order_delta(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get time between orders.

    Args:
    ----
    - df: input dataframe.

    Returns:
    --------
    - Dataframe with time between orders added.
    """
    _df = (
        df.lazy()
        .groupby("client_id")
        .agg(pl.col("date_order").unique())
        .explode("date_order")
        .with_columns(pl.col("date_order").diff().over("client_id").alias("order_time"))
        .drop_nulls()
    )

    return (
        df.lazy().join(
            other=_df,
            left_on=["client_id", "date_order"],
            right_on=["client_id", "date_order"],
        )
    ).collect()


def get_dynamic_features(df: pl.DataFrame, dt: str) -> pl.DataFrame:
    """
    Get features on a quarterly basis.

    Args:
    -----
    - df: input dataframe.

    Returns:
    --------
    - Dataframe with features on a quarterly basis added.
    """
    return (
        (
            df.lazy()
            .sort("date_order")
            .groupby_dynamic(
                "date_order", every=dt, by="client_id", include_boundaries=True
            )
            .agg(
                [
                    pl.col("sales_net").sum(),
                    pl.col("order_time").mean().dt.days(),
                    pl.col("order_channel").n_unique().alias("n_channels"),
                    pl.col("branch_id").n_unique().alias("n_branches"),
                    pl.col("product_id").n_unique().alias("n_products"),
                    pl.col("date_invoice").n_unique().alias("n_order"),
                    (pl.col("date_invoice") - pl.col("date_order"))
                    .mean()
                    .dt.days()
                    .alias("payment_time"),
                ]
            )
        )
        .collect()
        .drop(["_lower_boundary", "date_order"])
        .rename({"_upper_boundary": "date"})
    )


def get_rolling_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Get Rolling features on a quarterly basis.

    Args:
    -----
    - df: input dataframe.

    Returns:
    --------
    - Dataframe with rolling features on a quarterly basis added.
    """
    return (
        df.lazy()
        .sort("date")
        .with_columns(
            pl.col("sales_net").pct_change().over("client_id").alias("sales_change"),
            pl.col("order_time")
            .pct_change()
            .over("client_id")
            .alias("order_time_change"),
            pl.col("n_channels").pct_change().over("client_id").alias("channel_change"),
            pl.col("n_products").pct_change().over("client_id").alias("product_change"),
            pl.col("n_order").pct_change().over("client_id").alias("order_n_change"),
            pl.col("payment_time")
            .pct_change()
            .over("client_id")
            .alias("payment_time_change"),
        )
        .drop_nulls()
    ).collect()


def get_churn(df: pl.DataFrame, churn_threshhold: float) -> pl.DataFrame:
    """Create churn column based on sales change threshold from one quarter to the next.
    Args:
        df: input dataframe.
        churn threshold: threshold set for sales change.
    Returns:
        Dataframe with churn column added.
    """
    return (
        df.lazy()
        .sort("date")
        .with_columns(
            (
                pl.col("sales_change").shift(-1).over("client_id") < churn_threshhold
            ).alias("churn")
            * 1
        )
    ).collect()


def get_top_clients(df: pl.DataFrame, n_clients: int) -> pl.DataFrame:
    """
    Filter for top clients within a given period.

    Args:
    -----
    - df: input dataframe.
    - n_clients: number of clients to filter for.

    Returns:
    --------
    - Dataframe with top clients for each period.
    """
    return (
        df.lazy()
        .sort(["date", "sales_net"], reverse=True)
        .groupby("date", maintain_order=True)
        .head(n_clients)
    ).collect()


def get_static_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get static features.

    Args:
    -----
    - df: input dataframe.

    Returns:
    --------
    - Dataframe with static features.
    """
    return (
        df.lazy()
        .groupby("client_id")
        .agg(
            [
                pl.col("quali_relation").first(),
                pl.col("date_order").min().suffix("_first"),
                pl.col("date_order").max().suffix("_last"),
            ]
        )
    ).collect()


def merge_static(
    dynamic: pl.DataFrame,
    static: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge dataframes with static and dynamic features.

    Args:
    -----
    - dynamic: dataframe with dynamic features.

    Returns:
    --------
    - static: dataframe with static features.
    """
    df = (
        dynamic.lazy()
        .join(other=static.lazy(), left_on="client_id", right_on="client_id")
        .with_columns(
            [
                (pl.col("date") - pl.col("date_order_first"))
                .dt.days()
                .alias("client_age")
            ]
        )
        .filter(((pl.col("date_order_last") - pl.col("date"))) < timedelta(days=90))
        .drop(["date_order_first", "date_order_last"])
    ).collect()

    training = (
        df.lazy().filter(pl.col("date") < pl.col("date").max()).drop_nulls()
    ).collect()

    test = (
        df.lazy().filter(pl.col("date") == pl.col("date").max()).drop("churn")
    ).collect()

    return training, test


def data_pipeline(
    dt: str = config.period,
    churn_threshold: float = config.churn_threshhold,
    n_clients: int = config.n_clients,
    transaction_path: str = config.transactions_path,
    relations_path: str = config.relations_path,
) -> pl.DataFrame:
    """
    Full piepline for data loading and preproocessing.

    Args:
    -----
    - dt: span of churn periods.
    - churn_threshold: sales change threshold to determine churn.
    - n_clients: number of clients to filter by (top clients).
    - transaction_path: path for tran.
    - relation_path: path for client relation data.

    Returns:
    --------
    - df: preprocessed dataframe.
    """
    df = get_data(transaction_path=transaction_path, relation_path=relations_path)
    static = get_static_features(df)
    df = get_order_delta(df)
    df = get_dynamic_features(df, dt=dt)
    df = get_rolling_features(df)
    df = get_churn(df=df, churn_threshhold=churn_threshold)
    df = get_top_clients(df, n_clients=n_clients)
    return merge_static(dynamic=df, static=static)
