import streamlit as st

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.markdown("""<style> .dodge {background-color: dodgerblue; color: White;}</style>""",True)
st.markdown("""<h6 class="dodge">Nochmal ein kleiner Test</h6>""",True);
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.subheader("Learning to get used to Streamlit...")

#st.sidebar.header('User Input Parameters')


sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
features = pd.DataFrame(data, index=[0])
    


df = features


st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
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


st.subheader('Prediction Probability')
st.write(prediction_proba)
