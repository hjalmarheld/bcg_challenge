"""
This is where you put a model adopting the sklearn api

Final object must be named model
"""


from lightgbm import LGBMClassifier
import xgboost as xgb
import polars as pl

def recall(y_true, y_pred):
    data = {"churn": y_true, "predicted": y_pred}
    output = pl.DataFrame(data)
    result = (
        output.lazy()
        .with_columns([
            ((pl.col('churn')==pl.col('predicted')).alias('correct')*1),
            (((pl.col('churn')==pl.col('predicted')) & (pl.col('churn')==1)).alias('true_positive')*1),
            ((pl.col('churn')!=pl.col('predicted')) & (pl.col('predicted')==1)).alias('false_positive')*1,
            ((pl.col('churn')!=pl.col('predicted')) & (pl.col('predicted')==0)).alias('false_negative')*1])
    ).collect()
    recall = (
        result.select(pl.col('true_positive')).sum()
        / (result.select(pl.col('true_positive')).sum()+result.select(pl.col('false_negative')).sum())
        ).to_numpy()[0][0]
    return recall

model = LGBMClassifier(scale_pos_weight=3.2)
#model = xgb.XGBClassifier(objective=recall)