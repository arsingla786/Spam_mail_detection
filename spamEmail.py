#using logistic regression to detect spam emails 
#spam = 1 , not spam = 0 


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

df  = pd.read_csv('C:\\')

