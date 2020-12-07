# svm with class weight on an imbalanced classification dataset
import pandas as pd
from sklearn import svm

data = pd.read_csv('./data/featurize_data.csv')

# imbalanced data
# 20% seizure, 80% nonseizure
weights = {0: 0.8, 1: 0.2}

# define model
model = svm.SVC(kernel='linear', class_weight=weights)
