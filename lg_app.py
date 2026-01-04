import pandas as pd
import numpy as np
import sklearn
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder,StandardScaler
label_encoder=LabelEncoder()
scaler=StandardScaler()


model=pickle.load(open('logistic_model.pkl','rb'))

st.title('Logistic Regression for churn prediction')
gender=st.selectbox('Select Gender:',options=['Female','Male'])
SeniorCitizen=st.selectbox('You are a senior citizen?',options=['Yes','No'])
Partner=st.selectbox('Do you have partner?',options=['Yes','No'])
Dependents=st.selectbox('Are you dependens on other?',options=['Yes','No'])
tenure=st.text_input('Enter your tenure?')
PhoneService=st.selectbox('Do have phone Service?',options=['Yes','No'])
MultipleLines=st.selectbox('Do you have multilines service?',options=['Yes','No'])
Contract=st.selectbox('Your Contract:',options=['One year','Two year','Month-to-Month'])
TotalCharges=st.text_input("Enter your total charges?")


def predictive(gender,seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge):
    data={
    'gender':[gender],
    'SeniorCitizen':[seniorcitizen],
    'Partner':[Partner],
    'Dependents':[Dependents],
    'tenure':[tenure],
    'PhoneService':[Phoneservice],
    'MultipleLines':[multiline],
    'Contract':[contact],
    'TotalCharges':[totalcharge]
    }
    df1=pd.DataFrame(data)
    
    categorical_cols=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','Contract','TotalCharges']
    for column in categorical_cols:
        df1[column]=label_encoder.fit_transform(df1[column])
    df1=scaler.fit_transform(df1)
    result=model.predict(df1).reshape(1,-1)
    return result[0]


if st.button("Predict"):
    result=predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges)
    if result==0:
        st.write('Not Churn')
    else:
        st.write('Churn')