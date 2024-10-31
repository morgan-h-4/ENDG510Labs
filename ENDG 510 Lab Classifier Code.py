#load class
import numpy as np
import pandas as pd # in case it's not installed then install using conda create
import os
#load sklearn module for creating and evaluating ML models. In case sklearn isn't
from sklearn.neighbors import KNeighborsClassifier #load your classifier. In 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler #module for perform scaling
from sklearn.model_selection import train_test_split #module for splitting
from sklearn import metrics #module for evaluating performance

#load your data
df = pd.read_csv("data.csv") #change the name accordingly
df.head() # prints top 5 rows from the datatset to check data is load or not

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# remove duplicatesd
df = df.drop_duplicates()

# prepare features
x = df.drop(['Label'],axis=1) #remove class or label
y = df['Label'] #load label

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2) #split datatset. Here ratio is 80:20. Change accordingly

# Scale the data using standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) #scale training set
x_test = scaler.transform(x_test) #scale test set

z = KNeighborsClassifier(n_neighbors=3) # KNN classifier for 3 neighbours
zz = GaussianProcessClassifier()
zzz = GaussianNB()

KNN = z.fit(x_train,y_train) # start training
GPC = zz.fit(x_train,y_train) # start training
GNB = zzz.fit(x_train,y_train) # start training
predict_z = KNN.predict(x_test) # performance in the test set
predict_zz = GPC.predict(x_test) # performance in the test set
predict_zzz = GNB.predict(x_test) # performance in the test set


dict = {"Classifier Type":['KNN', 'GPC', 'GNB'],
        "Accuracy":[metrics.accuracy_score(y_test,predict_z), metrics.accuracy_score(y_test,predict_zz), metrics.accuracy_score(y_test,predict_zzz)]} # evaluating the
df = pd.DataFrame(dict)
print(df)

# library for save and load scikit-learn models
import pickle
# file name, recommending *.pickle as a file extension
filename = "model.pickle"
# save model
pickle.dump(z, open(filename, "wb"))