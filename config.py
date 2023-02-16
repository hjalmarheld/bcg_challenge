from pathlib import Path

# paths to datasets
transactions_path = Path("data", "transaction_data.parquet")
relations_path = Path("data", "sales_client_relationship_dataset.csv")
output_path = Path("data", "churn_predictions.csv")

# sales decrease to count as churn (standard = -0.5)
churn_threshhold = -0.5
# number of top clients to keep (standard = 10 000)
n_clients = 10_000
# span of churn periods (standard = 3mo)
period = "3mo"

# size of sliding window for training data
# -1 to include all data
window = -1

# save and load processed data as a pickle
cache = True

# simple logging of results in log.csv file
logging = True
