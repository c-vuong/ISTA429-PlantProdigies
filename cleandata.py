import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def datascale(weatherdf, yield_df):
    sc = MinMaxScaler(feature_range = (0, 1))
    X_train = []
    for i in range(len(weatherdf)):
        training_set_scaled = sc.fit_transform(weatherdf[i])
        X_train.append(np.array(training_set_scaled))
    # y_train is yield scaled 0-1
    yield_scaled = sc.fit_transform(yield_df)

    y_train = []
    X_train = np.array(X_train)

    y_train = np.array(yield_scaled[:200])

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],7))
    return X_train, y_train

def datascale2(weatherdf, yield_df):
    sc = MinMaxScaler(feature_range = (0, 1))
    X_train = []
    for i in range(len(weatherdf)):
        training_set_scaled = sc.fit_transform(weatherdf[i])
        X_train.append(np.array(training_set_scaled))
    # y_train is yield scaled 0-1
    yield_scaled = sc.fit_transform(yield_df)

    y_train = []
    X_train = np.array(X_train)

    y_train = np.array(yield_scaled[200:400])

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],7))
    return X_train, y_train


def buildModel(X_train, y_train):

    regressor = Sequential()

    regressor.add(LSTM(units = 1, return_sequences = True, input_shape = (214,7)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 1, return_sequences = True,))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 1))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

    return regressor

def saveModel(model):
    model.save('Model/')
    return

def graphModel(model,x_test,y_test):

    sc = MinMaxScaler(feature_range = (0, 1))
    sc.fit_transform(y_test)

    predicted_yield = model.predict(x_test)

    unscaled_predicted_yield = sc.inverse_transform(predicted_yield)
    unscaled_yield = []

    for i in predicted_yield:
        unscaled_yield.append(i)
    print(unscaled_yield)

    plt.plot(y_test, color = 'black', label = 'Actual Yield')
    plt.plot(unscaled_yield, color = 'green', label = 'Predicted Yield')
    plt.title('Yield Predictor')
    plt.xlabel('Record')
    plt.ylabel('Yield')
    plt.legend()
    plt.show()


    
    

def opendata():
    # For each record, daily weather data - a total of 214 days spanning the crop growing season (defined April 1 through October 31). Daily weather records were compiled based on the nearest grid point from a gridded 30km product. Each day is represented by the following 7 weather variables - 
    # Average Direct Normal Irradiance (ADNI)
    #
    # Average Relative Humidity (ARH)
    # Maximum Direct Normal Irradiance (MDNI)
    # Maximum Surface Temperature (MaxSur)
    # Minimum Surface Temperature (MinSur)
    # Average Surface Temperature (AvgSur)
    weather_data = np.load('inputs_weather_train.npy')
    # Maturity Group (MG), Genotype ID, State, Year, and Location for each performance record.
    other_data = np.load('inputs_others_train.npy')
    # Yearly crop yield value for each record.
    yield_data = np.load('yield_train.npy')
    
    df_weather = []
    df_weather_validation = []
    for i in range(0,200):
        df_weather.append(pd.DataFrame(weather_data[i]))
    for i in range(200,400):
        df_weather_validation.append(pd.DataFrame(weather_data[i]))
    
    df_yield = pd.DataFrame(yield_data)

    weather_titles = ["Average Direct Normal Irradiance (ADNI)", "Average Precipitation (AP)", "Average Relative Humidity (ARH)", "Maximum Direct Normal Irradiance (MDNI)", "Maximum Surface Temperature (MaxSur)", "Minimum Surface Temperature (MinSur)", "Average Surface Temperature (AvgSur)"]
    #for i in range(len(df_weather)):
    #    plt.plot(df_weather[i])
    #    plt.title(weather_titles[i])
    #    #plt.show()
    #print(df_weather)
    return df_weather, df_yield, df_weather_validation

def main():
    weather_df,yield_df, weather_validation = opendata()
    weather_df_scaled, yield_df_scaled = datascale(weather_df,yield_df)
    weather_valid_scaled, yield_valid_df = datascale2(weather_validation, yield_df)
    model = buildModel(weather_df_scaled,yield_df_scaled)
    saveModel(model)
    graphModel(model,weather_valid_scaled, yield_valid_df)


if __name__ == '__main__':
    main()