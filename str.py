import streamlit as st
st.title("Das ist eine Textzeile")
st.markdown("""<button>Ein Button</button>""",True)

st.markdown("""## Kleiner Versuch mit Markdown und HTML <br>
<button> Knopf zum Abschalten von Lärm </button>""", True)
st.markdown("""### Versuch hat bisher geklappt <br>
<form><label for='name'><input tye='text' id='name'>Name</label></form> """, True)

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# In[2]:


st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")


st.sidebar.header('User Input Parameters')


# In[4]:


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


# In[6]:


st.subheader('User Input parameters')
st.write(df)


# In[7]:


iris = datasets.load_iris()
X = iris.data
Y = iris.target


# In[8]:


clf = RandomForestClassifier()
clf.fit(X, Y)


# In[9]:


prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# In[10]:


st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)


# In[11]:


st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)


# In[12]:


st.subheader('Prediction Probability')
st.write(prediction_proba)
