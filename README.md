# MLCAS2021
This repo contains Plant Prodigies submission for the MLCAS 2021 Crop Yield Prediction Challenge

## Contributors
 - Alexis Parra
 - Manuel Perez
 - Michael Witusik
 - Amir Love
 - Christopher Vuong

## For each file in the repository 
cleandata.py -> This file has functions for changing the data to be use when training the model and output a graph of the model to help understand the actual yield and predicted yield<br><br>

## Model 
For our model we went with an LSTM which was originally used to predict stock prices, however, by updating the model we were able to make it predict yield data. 

## To run
To run the model all that is need is the cleandata.py in the same working directory as well as the data that can be found at link-> https://drive.google.com/file/d/1DoyextA0q4mxumMAhBvqZbfZriIM9A-Y/view
and then run the cleandata.py file, the program was tested with python 3.9.7

## Competition
Competition link -> https://eval.ai/web/challenges/challenge-page/1251/overview

## Acknowledgements 
Thank you to Team Data and Team X in ISTA 429! We gained a lot of knowledge by looking at your code. In addition, below is the website that we used to craft our LSTM model. Links:<br><br>
Team Data - https://github.com/Shadoweyes75/MLCAS2021 <br><br>
Team X - https://github.com/jake-newton/X429midterm<br><br>
LSTM Model Used- https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html<br><br>