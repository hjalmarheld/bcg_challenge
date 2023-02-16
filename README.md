# ClientCo Churn Prediction

The goal of this project is to create a churn prediction pipeline that will enable ClientCo to identify churners and take action. The next steps will be to help ClientCo prioritize the clients predicted to churn based on the value gained, in order to optimize their sales budget.


## 🚀 Getting started with the repository


- Firstly the data needs to be fetched and placed in the data folder, if needed update the paths in *config.py*.

- Secondly, get the required packages. The data pipeline is entirely *Polars* based and the base model is *LightGBM*, if needed, install the packages by running:

```
pip install polars lightgbm
```

- Lastly the model can be ran using the following commands from the source of the repo: 
 
```
python main.py
```

## 🗂 Repository structure

Our repository is structured in the following way:

```
|bcg_challenge
   |--data
   |-----data_pipeline.py
   |-----transaction data
   |-----relationship data
   |-----model output
   |--model
   |-----model_pipeline.py
   |-----model.py
   |--config.py
   |--main.py
   |--README.md
   |--logs.csv
```

### ❤️ Main.py
This python file serves as the main script and heart of the repository that runs the entire pipeline for the project.

### 🔢 config.py 
The configuration file where parameters are set.

### 🔢 logs.csv 
The logs.csv will contain the results of the different models trained.

## 📊 data
The Data folder contains the dataset used to train and test the model, prediction outputs and the data pipeline.

### data_pipeline.py
This file contains all the feature engineering and transformations done to prepare the training and test sets for the model.

## ℹ️ model
The folder containing the whole churn prediction model pipeline from data loading and preprocessing to training and predicting.

### model.py
This file contains the model architecture used.

### model_pipeline.py
This file contains the model pipeline: backtesting, live predictions.


## 📫 Contacts LinkedIn 

If you have any feedback, please reach out to us on LinkedIN!!!

- [Lea Chader](https://www.linkedin.com/in/lea-chader/)
- [Erik Held](https://www.linkedin.com/in/erik-held/)
- [Maxence Heitz](https://www.linkedin.com/in/maxenceheitz/)
- [Clemence Daurat](https://www.linkedin.com/in/cl%C3%A9mence-daurat/)
- [Delia Pouch](https://www.linkedin.com/in/d%C3%A9lia-pouch-antona/)
- [Martin Pointeaux](https://www.linkedin.com/in/martin-pointeaux-78a2391a2/)
