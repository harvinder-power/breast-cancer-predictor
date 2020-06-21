import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

st.sidebar.header("""
© Harvinder Power
# Input Parameters""")

classificationModel = st.sidebar.selectbox("Model", ["Random Forest Classifier", "GaussianNB", "K Nearest Neighbours", "Decision Tree Classifier"])
texture = st.sidebar.slider('Texture', 9.0, 40.0, 12.0)
perimeter = st.sidebar.slider('Perimeter', 40.0, 190.0, 100.0)
smoothness = st.sidebar.slider('Smoothness', 0.01, 0.18, 0.10)
compactness = st.sidebar.slider('Compactness', 0.005, 0.400, 0.200)


st.title('Breast Cancer Histology Prediction')

data = pd.read_csv('dataset.csv', header=0)

##Cleaning dataset
data.drop('id', axis=1, inplace = True)
data.drop("Unnamed: 32",axis=1,inplace=True)

features_mean = list(data.columns[1:11])
features_se = list(data.columns[11:20])
features_worst = list(data.columns[20:31])


## Mapping malignant to 1 and benign to 0
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})


## Preparing data for the model
prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
train, test = train_test_split(data, test_size = 0.3)
train_x = train[prediction_var]
train_y = train.diagnosis

test_x = test[prediction_var]
test_y = test.diagnosis

## Selecting model to use
if classificationModel == "Random Forest Classifier":
    model = RandomForestClassifier(n_estimators=100) 

if classificationModel == "GaussianNB":
    model = GaussianNB() 

if classificationModel == "K Nearest Neighbours":
    model = KNeighborsClassifier() 

if classificationModel == "Decision Tree Classifier":
    model = DecisionTreeClassifier() 

## Training the model and getting accuracy
model.fit(train_x, train_y)
prediction = model.predict(test_x)

accuracy = "Accuracy =", metrics.accuracy_score(prediction, test_y)
accuracy

prediction2 = model.predict([[texture, perimeter, smoothness, compactness]])

## Outputting prediction
st.header('Prediction')
st.write('Prediction of diagnosis based on 4 key criteria as determind by data analysis of the Wisconsin Breast Cancer Dataset. Change the Input Parameters in the sidebar to see the diagnosis change.')
st.write('**0 = Benign, 1 = Malignant**')
prediction2

##Plot heatmap
st.header('Correlation Heatmap')
st.write('Analysis of variables in in the dataset to determine potential correlations.')
st.write('_Correlation between variables such as radius and perimeter are due to the derivative nature of calculation (area is derived from the radius hence ~100% correlation, not 100% due to minor rounding errors.)_')
corr = data[features_mean].corr()
plt.figure(figsize =(14,14))
heatmap = sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')

st.pyplot()

## Plot Scatter Matrix
st.header('Scatter Matrix')
st.write('Scatter matrix to visualise data to split into benign and malignant. *Blue = benign, Red = malignant*')
color_function = {0: 'blue', 1: 'red'}
#0 = blue, 1 = red
colors = data['diagnosis'].map(lambda x: color_function.get(x))
pd.plotting.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15,15))
st.pyplot()