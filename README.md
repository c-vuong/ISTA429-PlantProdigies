# MLCAS2021
This repo contains Plant Prodigies submission for the MLCAS 2021 Crop Yield Prediction Challenge

## Contributors
 - Alexis Parra
 - Manuel Perez
 - Michael Witusik
 - Amir Love
 - Christopher Vuong

## Necessary files for running 
data_handle.py -> This file organizes the training and test data into a format readily available for use in our training and test models.<br>
<br>
model_handle.py -> This file builds the Decision Tree model and returns the NumPy prediction array used for the competition submission.
<br>
## Model 
This model is based on a Decision Tree regressor, which is simple to understand and implement. The decision tree regressor model takes in two arrays, the training samples and the class labels (validation) for the training samples. After being fitted to the model, the prediction can be called just after. This model does not have any obvious parameters to change, other than the training bias used for the model. This can be changed by modifying the data_set column used for training (x) in the DecisionTree function (the default is AvgSur, or Average Surface Temperature.

## Usage
To reproduce the NumPy yield prediction, first run the data_handle.py file to compress and organize the training and test data used in the model. Then, by running the model_handle.py file, the NumPy yield prediction file is produced.

## Competition
Competition link -> https://eval.ai/web/challenges/challenge-page/1251/overview

## Acknowledgements 
We would like to thank Team X in assisting our development in our LSTM model, as well as relying heavily on Team Data Dudes for their insight on Decision Trees regressor models.  In addition, below is the website that we used to craft our LSTM model and Decision Tree model.

Links:</br>
Team Data - https://github.com/Shadoweyes75/MLCAS2021<br>
Team X - https://github.com/jake-newton/X429midterm<br>
LSTM Model Used - https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html<br>
Decision Trees - https://scikit-learn.org/stable/modules/tree.html
