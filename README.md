# ClientCo Churn Prediction

The goal of this project is to create a churn prediction pipeline that will enable ClientCo to identify churners and take action. The next steps will be to help ClientCo prioritize the clients predicted to churn based on the value gained, in order to optimize their sales budget.


## 🚀 Getting started with the repository

To run the model go to the console and run following command: 
 
```
python main.py
```

You should be at the source of the repository structure (ie. natives_deephedging) when running the command.

## 🗂 Repository structure

Our repository is structured in the following way:

```
|bcg_challenge
   |--data
   |-----data_pipeline.py
   |-----transaction data
   |-----relationship data
   |--model
   |-----model_pipeline.py
   |-----model.py
   |--config.py
   |--main.py
   |--README.md
   |--requirements.txt
   |--logs.csv
```

### 📊 data
The Data folder contains the dataset used to train and test the model. It is also in this folder that the predictions are saved.

### 🔢 config.py 
The configuration file where parameters are set.

### 🔢 logs.csv 
The logs.csv will contain the results of the different models trained.

### ℹ️ src
The folder containing the whole churn prediction model pipeline from data loading and preprocessing to training and predicting.

### data_pipeline.py
This file contains all the feature engineering and transformations done to prepare the training and test sets for the model.

### model.py
This file contains the model architecture used.

### model_pipeline.py
This file contains the model pipeline: training, logging results and logging results.

### ❤️ Main.py

This python file serves as the main script and heart of the repository that runs the entire pipeline for the project.


## 📫 Contacts LinkedIn 

If you have any feedback, please reach out to us on LinkedIN!!!

- [Lea Chader](https://www.linkedin.com/in/lea-chader/)
- [Erik Held](https://www.linkedin.com/in/erik-held/)
- [Maxence Heitz](https://www.linkedin.com/in/maxenceheitz/)
- [Clemence Daurat](https://www.linkedin.com/in/cl%C3%A9mence-daurat/)
- [Delia Pouch](https://www.linkedin.com/in/d%C3%A9lia-pouch-antona/)
- [Martin Pointeaux](https://www.linkedin.com/in/martin-pointeaux-78a2391a2/)
