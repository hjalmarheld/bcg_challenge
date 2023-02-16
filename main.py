import csv
import config
import polars as pl
from src.data_pipeline import data_pipeline
from src.model_pipeline import model_pipeline
from pathlib import Path

if __name__ == "__main__":
    if config.cache:
        print('Cache active')
        if not Path('cache').exists():
            Path('cache').mkdir()
        if Path('cache', 'train.parquet').exists() and Path('cache', 'test.parquet').exists():
            print('Loading data from cache')
            train = pl.read_parquet(Path('cache', 'train.parquet'))
            test = pl.read_parquet(Path('cache', 'test.parquet'))
        else:
            print('Generating data set')
            train, test = data_pipeline()
            print('Saving dataset to disk')
            train.write_parquet(Path('cache', 'train.parquet'))
            test.write_parquet(Path('cache', 'test.parquet'))
    else:
        print('Cache not active, will generate data')
        training, test = data_pipeline()

    results, live_predictions = model_pipeline(
        training=train,
        test=test)

    print('Backtesting results')
    print(results)
    live_predictions.write_csv(config.output_path)

    if config.logging:
        print('Logging results')
        myFile = open('log.csv', 'a')
        writer = csv.writer(myFile)
        writer.writerow(results.values())
        myFile.close()