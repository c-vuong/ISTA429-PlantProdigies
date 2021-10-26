from data_handle import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

inputs_other = np.load('inputs_others_train.npy')
yield_train = np.load('yield_train.npy')
inputs_weather_train = np.load('inputs_weather_train.npy')
clusterID_genotype = np.load('clusterID_genotype.npy')

def DecisionTree(data,day):
    data_set = data
    data_set.head(12)
    y = data_set['yield']
    x = data_set['AvgSur']
    
    y = (y.to_numpy()).reshape(-1, 1)
    x = (x.to_numpy()).reshape(-1, 1)
    #['ADNI','AP','ARH','MDNI','MaxSur','MinSur',]
    test_size = 0.5
    seed = 5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    predictions = model.predict(x_test)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)
      
    test_other = np.load('inputs_others_test.npy')
    inputs_weather_test = np.load('inputs_weather_test.npy')

    sample_two = pull_one_day(inputs_weather_test,day,10337)

    sample_two = pd.DataFrame(sample_two)
    test_other = pd.DataFrame(test_other)

    test_data = group_weather_yield(sample_two,test_other)
    test_data = test_data['AvgSur']
    test_data = (test_data.to_numpy()).reshape(-1, 1)
    predictions = model.predict(test_data)
    
    return predictions

def main():
    prediction_each_day = []
    for i in range(213):
        sample_one = pull_one_day(inputs_weather_train,i,93028)
        yield_merge = merge_yield_other(inputs_other,yield_train)
        group_data = group_weather_yield(sample_one,yield_merge)
        prediction_each_day.append(DecisionTree(group_data,i))
    
    print(prediction_each_day[0])
    tmp = pd.DataFrame(prediction_each_day)
    tmp.shape

    prediction_over_time=[]
    for i in range(10337):
        prediction_over_time.append(tmp[i].mean())

    prediction_over_time = pd.DataFrame(prediction_over_time)
    np.save("prediction_file",(prediction_over_time.to_numpy()))
if __name__ == '__main__':
    main()