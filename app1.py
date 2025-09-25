import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

#load the trained models
model=tf.keras.models.load_model('regression_model.h5',compile=False)

#load the encoder and scaler
with open('label_encoder.pkl','rb')as file:
    label_encoder=pickle.load(file)

with open('one_hott.pkl','rb')as file:
    one_hott=pickle.load(file)

with open('scalerr.pkl','rb')as file:
    scalerr=pickle.load(file)

#streamlit app
#user input 
geography=st.selectbox('Geography',one_hott.categories_[0])
age=st.slider('Age',18,92)
gender=st.selectbox('Gender',label_encoder.classes_)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('NumOfProducts',1,4)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_Active_member=st.selectbox('IsActiveMember',[0,1])

#prepeare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_Active_member],
    'Exited':[exited]
})
input_data['Gender'] = label_encoder.transform([gender])[0]
##one hot encode geography
geo_encoder=one_hott.transform([[geography]]).toarray()
geo_encoder_df=pd.DataFrame(geo_encoder,columns=one_hott.get_feature_names_out(['Geography']))

#combine data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)


#scaled the data
input_data_scaled=scalerr.transform(input_data)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_salary=prediction[0][0]

st.write(f'Predicted estimated salary: ${prediction_salary:.2f}')
