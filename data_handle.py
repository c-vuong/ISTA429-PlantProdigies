import numpy as np
import pandas as pd
import pandas as pd
import numpy as np

#compress weather
def buildWeather(inputs_weather_train):
    new_weather = np.column_stack((np.repeat(np.arange(93028),214),inputs_weather_train.reshape(93028*214,-1)))
    new_weather = pd.DataFrame(new_weather, columns = ['PR','ADNI','AP','ARH','MDNI','MaxSur','MinSur','AvgSur'])
    return new_weather

#pulls one day out of weather file
def pull_one_day(dataWeather,day,ran):
    dayOne = []
    #dfw = pd.DataFrame(columns = ['AP','ARH','MDNI','MxSur','MnSur','MnSur','AgSur'])
    for i in range(ran):
        dayOne.append(dataWeather[i][day].tolist())
    #dfw.append(pd.DataFrame((dataWeather[i][0].tolist()) ,columns = ['AP','ARH','MDNI','MxSur','MnSur','MnSur','AgSur']))
    dfw = pd.DataFrame(dayOne,columns = ['ADNI','AP','ARH','MDNI','MaxSur','MinSur','AvgSur'])
    return dfw

#compress other data and yield
def merge_yield_other(inputs_other,yield_train):
    inputs_other_df = pd.DataFrame(inputs_other,columns = ['MG','GID','State','Year','loc'])
    yield_train_df = pd.DataFrame(yield_train,columns = ['yield'])
    inputs_other_df.insert(5,"yield", yield_train_df['yield'])
    return inputs_other_df

#makes the data smaller for testing
def make_data_small(dataFrame,size,random):
    if random:
        return dataFrame.sample(size)
    else:
        return dataFrame.head(size)

#brings weather and yields together
def group_weather_yield(weather,yields):
    return pd.concat([yields,weather], axis=1)