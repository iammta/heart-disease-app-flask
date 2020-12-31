import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import time
import warnings


warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 35)

print("\nModel developed by --> Mohammad Tariq")
print("Email id --> tariq9926097207@gmail.com")


# Exploratory analysis

data1 = pd.read_csv("HeartAttack_data.csv")
data1 = data1.replace("?", np.NaN)

print("\n\nHeart Attack data first five row and column::")
print(data1.head(5))

print("\nHeart Attack data Shape::")
print(data1.shape)

print("\nHeart Attack data Description")
print(data1.describe(include='all'))

print("\ndata1.num.unique()", data1.num.unique())

# data visualisation and pre-processing


plt.hist(data1['num'])
plt.title("num")
plt.show()

del data1['thal'] # deleting two column because of a lot of values are missing
del data1['ca']

# converting string column to float type and replacing missing values with median

data1['slope'] = pd.to_numeric(data1['slope'], downcast='float')
data1['slope'] = data1['slope'].fillna(data1['slope'].median())

data1['exang'] = pd.to_numeric(data1['exang'], downcast='float')
data1['exang'] = data1['exang'].fillna(data1['exang'].median())

data1['thalach'] = pd.to_numeric(data1['thalach'], downcast='float')
data1['thalach'] = data1['thalach'].fillna(data1['thalach'].median())

data1['restecg'] = pd.to_numeric(data1['restecg'], downcast='float')
data1['restecg'] = data1['restecg'].fillna(data1['restecg'].median())

data1['fbs'] = pd.to_numeric(data1['fbs'], downcast='float')
data1['fbs'] = data1['fbs'].fillna(data1['fbs'].median())

data1['chol'] = pd.to_numeric(data1['chol'], downcast='float')
data1['chol'] = data1['chol'].fillna(data1['chol'].median())

data1['trestbps'] = pd.to_numeric(data1['trestbps'], downcast='float')
data1['trestbps'] = data1['trestbps'].fillna(data1['trestbps'].median())


print("\ndata.groupby('num').size()-")
print(data1.groupby('num').size())

#we visualise the data using density plots to get a sense of the data distribution.

data1.plot(kind='density',subplots=True, layout=(3,4), sharex=False, legend=False, fontsize=1)
plt.show()



fig = plt.figure()
sub=fig.add_subplot(111)
#cax = sub.imshow(data1.corr() )
cax = sub.matshow(data1.corr(), vmin=-1, vmax=1)
sub.grid(True)
plt.title("Heart Disease Attributes")
fig.colorbar(cax)
plt.show()

# we'll split the data into predictor variables and target variable,

Y = data1['num'].values
X = data1.drop('num',axis=1).values


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=.33,random_state=123)

#Algorithm checks
#The following Algorithms will be checked
#1. Gaussian NB
#2. CART
#3. SVC
#4. KNN
#5. Logistic Regression


models=[]
models.append(('NB',GaussianNB()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('Logistic Regression',LogisticRegression()))

num_folds=10
results=[]
names=[]
print("\n\n")
for name,model in models:
    kfolds = KFold(n_splits=num_folds,random_state=123)
    startTime=time.time()
    cv_result=cross_val_score(model,X_train,Y_train,cv=kfolds,scoring='accuracy')
    endTime=time.time()
    results.append(cv_result)
    names.append(name)
    print("%s: %f (%f) (run time: %f)" % (name, cv_result.mean()*100, cv_result.std(), endTime-startTime))

#Results comparison using box-whisker graph
fig=plt.figure()
fig.suptitle("Result Comparison")
sub=fig.add_subplot(111)
plt.boxplot(results)
sub.set_xticklabels(names)
plt.show()


pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledLogistic Regression', Pipeline([('Scaler', StandardScaler()),('Logistic Regression', LogisticRegression())])))


results = []
names = []
print("\n\n\nAccuracies of Algorithms after Scaled dataset")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))



fig=plt.figure()
fig.suptitle("Result Comparison after Scaled Dataset")
sub=fig.add_subplot(111)
plt.boxplot(results)
sub.set_xticklabels(names)
plt.show()


# prepare the model

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

model = LogisticRegression()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\nLogistic Regression Training Completed. It's Run Time: %f" % (end-start))


# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by Logistic Regression Machine Learning Algorithms")
print("\n\nAccuracy score %f" % (accuracy_score(Y_test, predictions)))

print("\n")
print("Classification Report = \n")
print(classification_report(Y_test,predictions))


print("\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))

from sklearn.externals import joblib
filename =  "finalized_HeartAttack_data.sav"
joblib.dump(model, filename)
print( "\nBest Performing Model dumped successfully into a file by Joblib")

print("\n")
print("Model developed by --> Mohammad Tariq")
print("Email id --> tariq9926097207@gmail.com")
print(data1)
import pickle
filename =  "model_heart.pkl"
pickle.dump(model, open(filename,  "wb" ))
