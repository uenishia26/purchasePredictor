import numpy as np
from sklearn.svm import SVC #Suppoert vector Machine 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
from sklearn import metrics 

import pandas as pd 


df = pd.read_csv("/Users/anamuuenishi/Desktop/dataEntryEnv/practiceCSVML/Social_Network_Ads.csv")

X = df.iloc[:, [2,3]] 
y = df.iloc[:, [4]]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)


""" 
*** Basic Explanation of fit_transform & transform ***
fit_transfrom: Caluclates the sd and mean and other parameters to then transfrom the train data 
transfrom: Transforms the Xtest data to the same scale learnt from the training data 

Avoid overfitting: 
    fitting is only done on the Xtrain dataset. This is because in a real world scenario
    we must assume that the new data is unseen thus we cannot train the AI model using the
    unseen data. Thus, we must first fit on the training data and whatever statistics learnt
    from the trained data, we can then scale the testing data 
"""

stdScale = StandardScaler()
Xtrain = stdScale.fit_transform(Xtrain) #fit Calculates sd and mean / transfrom transforms all the data with the statistics
Xtest = stdScale.transform(Xtest) 


classifier = SVC(kernel='linear', random_state=0)
classifier.fit(Xtrain, ytrain)

#Predicting test results 
ypred = classifier.predict(Xtest)

print(metrics.accuracy_score(ytest, ypred)) #91% Accuracy score for linear kernel 

#Key note: While not implemented, the rbf kernel has a 93% accuracy score 


