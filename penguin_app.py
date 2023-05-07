import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
@st.cache
def predictions(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, gender):
  prediction = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, gender]])
  prediction = prediction[0]
  if prediction==0:
    return "Adelie"
  elif prediction == 1:
    return "Chinstrap"
  else:
    return "Gentoo"
st.title("Penguin Species Prediction App")
st_bill_length = st.sidebar.slider("Bill Length MM", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
st_bill_depth = st.sidebar.slider("Bill Depth MM", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
st_flipper_length = st.sidebar.slider("Flipper Length MM", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
st_body = st.sidebar.slider("Body Mass G", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))
st_gender = st.sidebar.selectbox("Gender:", ("Female", "Male"))
st_island = st.sidebar.selectbox("Island:", ("Biscoe", "Dream", "Torgersen"))
classifier = st.sidebar.selectbox("Classifier:", ("Support Vector Machine", "Logistic Regression", "RandomForestClassifier"))
if st_gender == "Male":
	st_gender = 0
else:
	st_gender = 1
if st_island == "Biscoe":
	st_island = 0
elif st_island == "Dream":
	st_island = 1
else:
	st_island = 2
if st.sidebar.button("Predict"):
  if classifier =="Support Vector Machine":
    species_type = predictions(svc_model, st_island, st_bill_length, st_bill_depth, st_flipper_length, st_body, st_gender)
    score = svc_model.score(X_train, y_train)
  elif classifier == "Logistic Regression":
    species_type = predictions(log_reg, st_island, st_bill_length, st_bill_depth, st_flipper_length, st_body, st_gender)
    score = log_reg.score(X_train, y_train)
  else:
    species_type = predictions(rf_clf, st_island, st_bill_length, st_bill_depth, st_flipper_length, st_body, st_gender)
    score = rf_clf.score(X_train, y_train)
  st.write("Penguin Species Predicted: ", species_type)
  st.write("Model Score Accuracy: ", score)