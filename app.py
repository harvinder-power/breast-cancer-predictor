import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Breast Cancer Prediction
This web app predicts risk of breast cancer from the Wisconsin Breast Cancer Dataset.
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    radius_mean = st.sidebar.slider('Radius Mean', 6.0, 30.0, 8.0)
    texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 12.0)
    perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 190.0, 100.0)
    area_mean= st.sidebar.slider('Area Mean', 140.0, 2550.0, 700.0)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.01, 0.18, 0.10)
    compactness_mean = st.sidebar.slider('Compactness Mean', 0.005, 0.400, 0.200)
    concavity_mean = st.sidebar.slider('Concavity Mean', 0.00, 0.44, 0.25)
    concave_points_mean	= st.sidebar.slider('Concave Points Mean', 0.00, 0.24, 0.14)
    symmetry_mean = st.sidebar.slider('Symmetry Mean', 0.08, 0.35, 0.20)
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', 0.02, 0.10, 0.05)
    radius_se = st.sidebar.slider('Radius SE', 0.10, 2.90, 1.30)
    texture_se = st.sidebar.slider('Texture SE', 0.35, 5.00, 3.50)
    perimeter_se= st.sidebar.slider('Perimeter SE', 0.70, 22.5, 12.0)
    area_se	= st.sidebar.slider('Area SE', 5.0, 550.0, 15.0)
    smoothness_se = st.sidebar.slider('Smoothness SE', 0.000, 0.040, 0.010) 
    compactness_se = st.sidebar.slider('Compactness SE', 0.0001, 0.1500, 0.0090)
    concavity_se = st.sidebar.slider('Concavity SE', 0.00, 0.40, 0.20)
    concave_points_se = st.sidebar.slider('Concave Points SE', 0.00, 0.06, 0.02)
    symmetry_se = st.sidebar.slider('Symmetry SE', 0.006, 0.080, 0.020)
    fractal_dimension_se = st.sidebar.slider('Fractal Dimension SE', 0.000, 0.030, 0.015)
    radius_worst = st.sidebar.slider('Radius Worst', 7.0, 37.0, 12.0)
    texture_worst = st.sidebar.slider('Texture Worst', 11.5, 50.0, 31.0)
    perimeter_worst	= st.sidebar.slider('Perimeter Worst', 50.0, 255.0, 122.0)
    area_worst = st.sidebar.slider('Area Worst', 180.0, 4500.0, 700.0)
    smoothness_worst = st.sidebar.slider('Smoothness Worst', 0.070, 0.250, 0.140)
    compactness_worst = st.sidebar.slider('Compactness Worst', 0.025, 1.000, 0.21)
    concavity_worst = st.sidebar.slider('Concavity Worst', 0.0, 1.3, 0.6)
    concave_points_worst = st.sidebar.slider('Concave Points Worst', 0.00, 0.300, 0.10)
    symmetry_worst = st.sidebar.slider('Symmetry Worst', 0.150, 0.690, 0.420)
    fractal_dimension_worst = st.sidebar.slider('Fractal Dimension Worst', 0.050, 0.210, 0.140)



    data = {'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'perimeter_mean': perimeter_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'concave_points_mean': concave_points_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'perimeter_se': perimeter_se,
            'area_se': area_se,
            'smoothness_se':smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se':concavity_se,
            'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'radius_worst': radius_worst,
            'texture_worst': texture_worst,
            'perimeter_worst': perimeter_worst,
            'area_worst': area_worst,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'concave_points_worst': concave_points_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst
            }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_breast_cancer()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)