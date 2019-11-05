import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("insurance.csv")

# check the any is missing
df.isnull().sum()


#from sklearn.preprocessing import LabelEncoder 
#le = LabelEncoder()
#x.sex = le.fit_transform(x.sex)
#x.smoker = le.fit_transform(x.smoker)


#data preprocessing
dm1 = pd.get_dummies(df.sex)
merg1 = pd.concat([dm1,df],axis = 'columns')

dm2 = pd.get_dummies(df.smoker)
merg2 = pd.concat([dm2,merg1],axis = 'columns')

dm3 = pd.get_dummies(df.region)
merg3 = pd.concat([dm3,merg2],axis = 'columns')

final = merg3.drop(['northeast','sex','smoker','region'],axis = 'columns')

# spliting the data
x = final.iloc[:, 0:10]
y = final.iloc[:, 10]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#train the regression models
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# prediction
regressor.predict(x_test)

# score of models
regressor.score(x_train, y_train)


# +++++++SAVE THE MODEL+++++ 
#(1st METHODS)
import pickle

# save
with open('regressor_pickle','wb') as f:
    pickle.dump(regressor,f)
    
#retrive the model
with open('regressor_pickle','rb') as f:
    mp = pickle.load(f)
       
#use again the model
mp.predict(x_test)

#+++++++SAVE THE MODEL+++++++++
#this model useful for when we have lots of array
#(2nd method)
from sklearn.externals import joblib

#save
joblib.dump(regressor,'reg_joblib')

#retrive the model
ab = joblib.load('reg_joblib')

# use agian the model
ab.predict(x_test)
