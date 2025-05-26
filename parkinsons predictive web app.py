import numpy as np
import pickle
import streamlit as st

scaler = pickle.load(open("E:/python/ML Projects/14 - Parkinson's Disease Detection using Machine Learning/scaler.sav", 'rb'))
model = pickle.load(open("E:/python/ML Projects/14 - Parkinson's Disease Detection using Machine Learning/parkinsons_model.sav", 'rb'))


def parkinsons_prediction(data):
    data_array = np.fromstring(data, sep=",")
    input_reshaped = data_array.reshape(1,-1)
    std_data = scaler.transform(input_reshaped)
    prediction = model.predict(std_data)

    if prediction[0] == 1:
        return 'The Person does not have Parkinsons Disease'
    else:
        return 'The Person has Parkinsons Disease'


# The  main func
def main():

    st.title('Parkinsons Disease Prediction Web App')

    input_data = st.text_input('Input the values of the features in the order:')

    diagnosis = ''

    if st.button('parkinsons prediction'):
        diagnosis = parkinsons_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()