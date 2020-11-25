import pandas as pd


def read_csv(csv_filename):
    # read data file
    data = pd.read_csv(csv_filename)

    # remove unnecessary columns
    # data.drop(data.columns[0], axis=1, inplace=True)

    # assign binary label 
    data['label'] = (data.y == 1).astype(int)

    # FEATURES
    features = pd.DataFrame()
    features['min'] = data.min(axis=1)
    features['max'] = data.max(axis=1)
    features['mean'] = data.mean(axis=1) 
    features['median'] = data.median(axis=1) 
    #features['mode'] = data.mode(axis=1) 
    features['std'] = data.std(axis=1)

    # line length 
    

    # entropy 

    # energy 
    features['energy'] = data.sum(data**2, axis=1)

    # peak frequency

    return features


data = read_csv("./data.csv")
print(data)