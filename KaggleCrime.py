import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training = pd.read_csv("Train.csv")
testing = pd.read_csv("Test.csv")

def DateTimeStamp(data):
    dates = []
    time = []
    for iter in data:
        dateimp = iter.split(" ")
        datestamp = dateimp[0]
        timestamp = dateimp[1]
        dates.append(datestamp)
        time.append(timestamp)
    return dates,time

col1,col2 = DateTimeStamp(training.Dates)
col3,col4 = DateTimeStamp(testing.Dates)

def OriginalInfo(dat1,dat2):
    typ1 = []
    typ2 = []
    typ3 = []
    maintime = []
    for iter in dat1:
        dat = iter.split("-")
        sub1 = dat[0]
        sub2 = dat[1]
        sub3 = dat[2]
        typ1.append(sub1)
        typ2.append(sub2)
        typ3.append(sub3)
    for row in dat2:
        time = row.split(":")
        main = time[0]
        maintime.append(main)
    return typ1,typ2,typ3,maintime

colu1,colu2,colu3,colu4 = OriginalInfo(col1,col2)        
colum1,colum2,colum3,colum4 = OriginalInfo(col3,col4)

del col1,col2,col3,col4

training["Year"] = colu1
training["Month"] = colu2
training["Date"] = colu3
training["Hour"] = colu4

del training["Dates"]
del training["Descript"],training["Resolution"]

testing["Year"] = colum1
testing["Month"] = colum2
testing["Date"] = colum3
testing["Hour"] = colum4

del testing["Dates"]
del training["Address"],testing["Address"]

del colu1,colu2,colu3,colum1,colum2,colum3,colu4,colum4

Y = training.iloc[:, 0].values
X = training.iloc[:, 1:].values
test_data = testing.iloc[:, 1:].values    

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
code1 = LabelEncoder()
X[:, 0] = code1.fit_transform(X[:, 0])
test_data[:, 0] = code1.transform(test_data[:, 0])
code2 = LabelEncoder()
X[:, 1] = code2.fit_transform(X[:, 1])
test_data[:, 1] = code2.transform(test_data[:, 1])
code3 = LabelEncoder()
Y = code3.fit_transform(Y)
hot = OneHotEncoder(categorical_features = [0,1,4,5,6,7])
X = hot.fit_transform(X).toarray()
hot2 = OneHotEncoder(categorical_features = [0,1,4,5,6,7])
test_data = hot2.fit_transform(test_data).toarray()

X = np.delete(X, [0,7,17,30,42,73],axis = 1)
test_data = np.delete(test_data, [0,7,17,30,42,73],axis = 1)

from sklearn.decomposition import PCA
pca = PCA(n_components=7)
X = pca.fit_transform(X)
test_data = pca.transform(test_data)
explained_variance = pca.explained_variance_ratio_

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X,Y)

y_pred = classifier.predict(test_data)

import collections
collections.Counter(y_pred)
collections.Counter(Y)


from keras.utils.np_utils import to_categorical
y_pred = to_categorical(y_pred)

df = pd.DataFrame({'Id': testing.Id,'ARSON': y_pred[:, 0],'ASSAULT' : y_pred[:, 1],'BAD CHECKS': y_pred[:, 2],'BRIBERY': y_pred[:, 3],'BURGLARY': y_pred[:, 4],'DISORDERLY CONDUCT': y_pred[:, 5],'DRIVING UNDER THE INFLUENCE': y_pred[:, 6],'DRUG/NARCOTIC': y_pred[:, 7],'DRUNKENNESS': y_pred[:, 8],'EMBEZZLEMENT': y_pred[:, 9],'EXTORTION': y_pred[:,10],'FAMILY OFFENSES': y_pred[:, 11],'FORGERY/COUNTERFEITING': y_pred[:, 12],'FRAUD': y_pred[:, 13],'GAMBLING': y_pred[:, 14],'KIDNAPPING': y_pred[:,15],'LARCENY/THEFT': y_pred[:, 16],'LIQUOR LAWS': y_pred[:, 17],'LOITERING': y_pred[:, 18],'MISSING PERSON': y_pred[:, 19],'NON-CRIMINAL': y_pred[:, 20],'OTHER OFFENSES': y_pred[:, 21],'PORNOGRAPHY/OBSCENE MAT': y_pred[:, 22],'PROSTITUTION': y_pred[:, 23],'RECOVERED VEHICLE': y_pred[:, 24],'ROBBERY': y_pred[:, 25],'RUNAWAY': y_pred[:, 26],'SECONDARY CODES': y_pred[:, 27],'SEX OFFENSES FORCIBLE': y_pred[:, 28],'SEX OFFENSES NON FORCIBLE': y_pred[:, 29],'STOLEN PROPERTY': y_pred[:, 30],'SUICIDE': y_pred[:, 31],'SUSPICIOUS OCC':y_pred[:, 32],'TREA': y_pred[:, 33],'TRESPASS': y_pred[:, 34],'VANDALISM': y_pred[:, 35],'VEHICLE THEFT': y_pred[:, 36],'WARRANTS': y_pred[:, 37],'WEAPON LAWS': y_pred[:, 38]},columns = ['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])
df.to_csv('D:\Movies\KaggleCrime.csv',sep = ',',header=True,index=False)
