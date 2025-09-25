import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

##load ht etrained model
model=tf.keras.models.load_model('model.h5')

with open('label_encode.pkl','rb')as file:
    label_encode=pickle.load(file)

with open('one_hot.pkl','rb')as file:
    one_hot=pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

## streamlit app
st.title("Customer churn prediction")

#user input 
geography=st.selectbox('Geography',one_hot.categories_[0])
age=st.slider('Age',18,92)
gender=st.selectbox('Gender',label_encode.classes_)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('EstimatedSalary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('NumOfProducts',1,4)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_Active_member=st.selectbox('IsActiveMember',[0,1])

#prepeare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encode.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_Active_member],
    'EstimatedSalary':[estimated_salary]
})
input_data['Gender'] = label_encode.transform([gender])[0]
##one hot encode geography
geo_encoder=one_hot.transform([[geography]]).toarray()
geo_encoder_df=pd.DataFrame(geo_encoder,columns=one_hot.get_feature_names_out(['Geography']))

#combine data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)


#scaled the data
input_data_scaled=scaler.transform(input_data)

#predict churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
    st.write("The customer is likely to churn")

else:
    st.write("The customer is not likely to churn")