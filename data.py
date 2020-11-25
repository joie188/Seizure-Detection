import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.fft import fft

def read_csv(csv_filename):
    # read data file
    data = pd.read_csv(csv_filename)

    # remove unnecessary columns
    # data.drop(data.columns[0], axis=1, inplace=True)

    # assign binary label 
    #data['label'] = (data.y == 1).astype(int)
    
    data_np = data.to_numpy()
    data_np = data_np[:, 1:-1].astype(float)
    #print(data_np)

    # FEATURES
    features = pd.DataFrame()
    # features['min'] = np.min(data_np, axis=1)
    # features['max'] = np.max(data_np, axis=1)
    # features['mean'] = np.mean(data_np, axis=1) 
    # features['median'] = np.median(data_np, axis=1) 
    # features['std'] = np.std(data_np, axis=1)

    # # line length 
    # L = data_np[:,1:] - data_np[:,:-1]
    # L = np.sum(np.abs(L), axis=1)
    # features['line_length'] = L

    # entropy = []
    # for row in range(10):
    #     r = data_np[row,:]
    #     #entropy.append(ent.sample_entropy(r, 2, 0.2 * np.std(r)))
    #     entropy.append(sampen(r, 2, 0.2*np.std(r)))
    # print(entropy)
    # features['entropy'] = entropy

    # # energy 
    # features['energy'] = np.sum(data_np**2, axis=1)

    # peak frequency
    x = fft(data_np)
    x = np.absolute(x)
    
    coeff = np.argmax(x, axis=1)
    features['peak_f'] = coeff

    median_fs = []
    for row in x:
        total_sum = np.sum(row)
        running_sum = 0 
        for i in range(len(row)):
            running_sum += row[i] 
            if running_sum >= total_sum/2:
                break 
        median_fs.append(i)
    features["median_f"] = median_fs


    return features

def sampen(L, m, r):
    N = len(L)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    
    print(xmi[0].shape)
    print("Help:", np.shape(np.abs(xmi[0] - xmj).max(axis=1)))

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


data = read_csv("./data.csv")
print(data)