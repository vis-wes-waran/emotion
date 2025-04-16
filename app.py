import streamlit as st
import pickle

with open('TF','rb') as f:
      TF=pickle.load(f)

with open('LabelEncoder','rb') as f:
     label=pickle.load(f)

with open('language_detection','rb') as f:
     model=pickle.load(f)


st.title("Language Detector")

user_input = st.text_input("Enter Language")

if st.button('Detection'):

    vector=TF.transform([user_input])

    predict=model.predict(vector)

    st.success(label.inverse_transform(predict))
    
