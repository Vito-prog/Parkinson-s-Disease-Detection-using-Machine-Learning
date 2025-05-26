import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm

scaler = pickle.load(open("E:/python/ML Projects/14 - Parkinson's Disease Detection using Machine Learning/scaler.sav", 'rb'))
model = pickle.load(open("E:/python/ML Projects/14 - Parkinson's Disease Detection using Machine Learning/parkinsons_model.sav", 'rb'))

input = [202.26600,211.60400,197.07900,0.00180,0.000009,0.00093,0.00107,0.00278,0.00954,0.08500,0.00469,0.00606,0.00719,0.01407,0.00072,32.68400,0.368535,0.742133,-7.695734,0.178540,1.544609,0.056141]
input_as_numpy_array = np.asarray(input)
input_reshaped = input_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_reshaped)
prediction = model.predict(std_data)
print(prediction[0])
if prediction[0] == 1:
    print('The Person does not have Parkinsons Disease')
else:
    print('The Person has Parkinsons Disease')