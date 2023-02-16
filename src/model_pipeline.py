import config
from src.model import model
import polars as pl

def model_pipeline(
        training: pl.DataFrame,
        test: pl.DataFrame,
        model=model,
        window=config.window,
        ):
    """Back testing and predictions on live data.
    Args:
        training: the training set.
        test: the testing set.
        model: the model architecture defined in model.py.
        window: the window for back testing.
    Returns:
        results of model and predictions.
    """
    time_points = training.sort('date').select(pl.col('date')).unique()[:, 0]

    # the first part of this function enables back testing of models, the second part does
    # predictions on live data

    # back testing
    assert window<time_points.shape[0]-1, "window too large, > %s" % (time_points.shape[0]-1)

    if window==-1:
        static=True
        window=1
    else:
        static=False

    output = []
    for t in range(window, time_points.shape[0]):
        predict_date = time_points[t]
        if static:
            start_date = time_points[0]
        else:
            start_date = time_points[t-window]
        print("Backtesting for :")
        print(predict_date)

        x_train = (
            training.lazy()
            .select(pl.all().exclude('churn'))
            .filter(pl.col('date')>=start_date)
            .filter(pl.col('date')<predict_date)
            .sort('client_id')
            .drop(['date','client_id'])
        ).collect().to_numpy()

        y_train = (
            training.lazy()
            .filter(pl.col('date')>=start_date)
            .filter(pl.col('date')<predict_date)
            .sort('client_id')
            .select(pl.col(['churn']))
        ).collect()[:, 0].to_numpy()

        x_predict = (
            training.lazy()
            .select(pl.all().exclude('churn'))
            .filter(pl.col('date')==predict_date)
            .sort('client_id')
            .drop(['date','client_id'])
        ).collect().to_numpy()

        y_actual = (
            training.lazy()
            .filter(pl.col('date')==predict_date)
            .select(pl.col(['client_id', 'date', 'churn']))
            .sort('client_id')
        ).collect()

        model.fit(x_train, y_train)
        predictions = model.predict(x_predict)
        y_actual = y_actual.with_columns(pl.Series(predictions).alias('predicted'))
        output.append(y_actual)

    result = (
        pl.concat(output).lazy()
        .with_columns([
            ((pl.col('churn')==pl.col('predicted')).alias('correct')*1),
            (((pl.col('churn')==pl.col('predicted')) & (pl.col('churn')==1)).alias('true_positive')*1),
            ((pl.col('churn')!=pl.col('predicted')) & (pl.col('predicted')==1)).alias('false_positive')*1,
            ((pl.col('churn')!=pl.col('predicted')) & (pl.col('predicted')==0)).alias('false_negative')*1])
    ).collect()

    accuracy = result.select(pl.col('correct')).mean().to_numpy()[0][0]
    precision = (
        result.select(pl.col('true_positive')).sum()
        / (result.select(pl.col('true_positive')).sum() + result.select(pl.col('false_positive')).sum())
        ).to_numpy()[0][0]
    recall = (
        result.select(pl.col('true_positive')).sum()
        / (result.select(pl.col('true_positive')).sum()+result.select(pl.col('false_negative')).sum())
        ).to_numpy()[0][0]

    results = {'model': str(model), 'accuracy':accuracy, 'precision':precision, 'recall':recall}


    # live data

    if static:
        start_date = time_points[0]
    else:
        start_date = time_points[-window]

    x_train = (
        training.lazy()
        .select(pl.all().exclude('churn'))
        .filter(pl.col('date')>=start_date)
        .sort('client_id')
        .drop(['date','client_id'])
    ).collect().to_numpy()

    y_train = (
        training.lazy()
        .filter(pl.col('date')>=start_date)
        .sort('client_id')
        .select(pl.col(['churn']))
    ).collect()[:, 0].to_numpy()

    print("Making live predictions")
    x_real = test.select(pl.all().exclude(['date', 'client_id'])).to_numpy()
    model.fit(x_train, y_train)

    live_predictions = (
        test.select(pl.col(['client_id']))
        .with_columns(pl.Series(model.predict(x_real)).alias('predicted_churn'),
                      pl.Series(model.predict_proba(x_real)[:,1]).alias('predicted_churn_proba'))
    )

    return results, live_predictions