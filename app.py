import tensorflow
import streamlit as st
import pandas as pd

from tensorflow.keras.models import load_model
#load pickle files
import pickle
model=load_model('model.h5')
with open('label_encode_gender.pickle','rb') as file:
  label_encode_gender =pickle.load(file)
with open('onehot_encode_geo.pickle','rb') as file:
  onehot_encode_geo = pickle.load(file)

st.title('churn Prediction')
geography=st.selectbox('Geography',onehot_encode_geo.categories_[0])
gender = st.selectbox("Gender",['Male','Female'])
age=st.slider('Age',18,92)
balance=st.number_input('Balance',min_value=0.0)
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox('HasCredit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encode_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])



geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encode_geo.get_feature_names_out(['Geography']))
input_full = pd.concat([input_df,geo_encoded_df],axis=1)

#scaling input features
with open('scaler.pickle','rb') as file:
  scaler=pickle.load(file)
# input_scaled=scaler.transform(input_df)
#scale the input data
input_data_scaled=scaler.transform(input_full)



#making Prediction
prediction=model.predict(input_data_scaled)
prediction_probe=prediction[0][0]
prediction_probe


if prediction_probe > 0.5:
  st.write("customer is likely churned")
else:
  st.write("customer not likely churned")