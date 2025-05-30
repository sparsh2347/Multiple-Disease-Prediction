import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu 

#loading the saved models
diabetes_model=pickle.load(open('C:/Users/spars/OneDrive/Desktop/Machine Learning/ML Projects/Multiple Disease Prediction/Diabetes/diabetes.sav','rb'))
heart_model=pickle.load(open('C:/Users/spars/OneDrive/Desktop/Machine Learning/ML Projects/Multiple Disease Prediction/Heart/heart.sav','rb'))
parkinson_model=pickle.load(open('C:/Users/spars/OneDrive/Desktop/Machine Learning/ML Projects/Multiple Disease Prediction/Parkinson/parkinson.sav','rb'))

#looding the scalar 
diabetes_scalar=pickle.load(open('C:/Users/spars/OneDrive/Desktop/Machine Learning/ML Projects/Multiple Disease Prediction/Diabetes/diabetes_scaler.pkl','rb'))
parkinson_scalar=pickle.load(open('C:/Users/spars/OneDrive/Desktop/Machine Learning/ML Projects/Multiple Disease Prediction/Parkinson/parkinson_scalar.pkl','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    data=input_data
    #converting the data to numpy array
    input_data_array=np.asarray(data)
    print("Input array:", input_data_array)
    print("Data types in array:", input_data_array.dtype)
    #reshaping the array as we are predicting for one instance
    input_data_reshaped=input_data_array.reshape(1,-1)
    input_scaled = diabetes_scalar.transform(input_data_reshaped)
    prediction=diabetes_model.predict(input_scaled)
    print(prediction)
    if(prediction==0):
      return "The person is not diabetic"
    else:
      return "The person is diabetic"

def heart_disease_prediction(input_data):
    input_arr=np.asarray(input_data,dtype=float)
    #reshaping the array for a single instance
    input_arr_reshaped=input_arr.reshape(1,-1)
    #predicting for the input based on trained model
    prediction=heart_model.predict(input_arr_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return 'The person does not have heart disease'
    else:
        return 'The person has heart disease'

def parkinsons_prediction(input_data):
    input_arr=np.asarray(input_data,dtype=float)
    #reshaping the array for a single instance
    input_arr_reshaped=input_arr.reshape(1,-1)
    #predicting for the input based on trained model
    prediction=parkinson_model.predict(input_arr_reshaped)
    print(prediction)
    if(prediction[0]==0):
        return 'The person does not have parkinsons disease'
    else:
        return 'The person has parkinsons disease'

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

    
#sidebar for navigate
with st.sidebar:
    #sidebar name:(Multiple Disease Prediction)
    # - web pages names
    #streamlit allows bootstrap icons
    selected=option_menu('Multiple Disease Prediction System',# Menu Name
                         ['Home','Diabetes Prediction','Heart Disease Prediction','Parkinsons Disease Prediction','Dashboard'],# Menu Items
                         icons=['file-medical','activity','heart-pulse','person','bar-chart'],# Menu Icons
                         default_index=0)
#deafult index 0 i.e when you will open the app diabetes prediction page will be the deafult page i.e it will open automatically
if (selected == 'Home'):
    st.title("Welcome to the Multiple Disease Prediction System")
    st.markdown("""
    This application predicts the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** 
    using machine learning models trained on medical datasets.

    ### Features:
    - Easy input interface
    - Accurate predictions
    - Save results to dashboard
    - Secure and offline
    """)
#Diabetes Preciction Page
elif(selected=='Diabetes Prediction'):
    #page title
    st.title("Diabetes Prediction")
     
    #creating input fields and  getting input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    #columns for input fields
    #  Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Pregnancies", help="Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose", help="Glucose Level")
    with col3:
        BloodPressure = st.text_input("BP", help="Blood Pressure Value")

    # Row 2
    with col1:
        SkinThickness = st.text_input("Skin", help="Skin Thickness Level")
    with col2:
        Insulin = st.text_input("Insulin", help="Insulin Level")
    with col3:
        BMI = st.text_input("BMI", help="Body Mass Index")

    # Row 3
    with col1:
        DiabetesPedigreeFunction = st.text_input("DPF", help="Diabetes Pedigree Function")
    with col2:
        Age = st.text_input("Age", help="Age of the person")
    
    #code for prediction
    diab_diagnosis=''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        #[[ ]] as we are predicting only for one person
        data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diab_diagnosis = diabetes_prediction([data])

        st.session_state.prediction_history.append({
            "Disease": "Diabetes",
            "Input": data,
            "Result": diab_diagnosis
        })
    st.success(diab_diagnosis)

#Heart Disease Prediction page
elif(selected=='Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction')

    #creating input fields and  getting input data from the user
    #Age,Sex,Chest Pain,Resting Blood Pressure,Cholestrol,FBS,resting electrocardiographic results,maximum heart rate achieved(thalach),exercise induced angina,Oldpeak,Slope,CA,Thal
    
    
    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input("Age", help="Age of the person")
    with col2:
        Sex = st.text_input("Sex", help="1 = male, 0 = female")
    with col3:
        CP = st.text_input("Chest Pain", help="Chest Pain Type (0–3)")

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        restbp = st.text_input("Rest BP", help="Resting Blood Pressure")
    with col5:
        chol = st.text_input("Cholesterol", help="Cholesterol Level")
    with col6:
        FBS = st.text_input("Fasting Sugar", help="1 = true, 0 = false")

    # Row 3
    col7, col8, col9 = st.columns(3)
    with col7:
        restecg = st.text_input("ECG", help="Resting Electrocardiographic Results (0–2)")
    with col8:
        thalach = st.text_input("Max HR", help="Maximum Heart Rate Achieved")
    with col9:
        exang = st.text_input("Angina", help="Exercise Induced Angina (1 = yes, 0 = no)")

    # Row 4
    col10, col11, col12 = st.columns(3)
    with col10:
        oldpeak = st.text_input("Oldpeak", help="ST Depression Induced by Exercise")
    with col11:
        slope = st.text_input("Slope", help="Slope of the ST Segment (0–2)")
    with col12:
        CA = st.text_input("Vessels", help="No. of Major Vessels Colored by Fluoroscopy (0–3)")

    # Row 5
    col13, _, _ = st.columns(3)
    with col13:
        thal = st.text_input("Thal", help="1 = normal, 2 = fixed defect, 3 = reversible defect")

    heart_diagnosis=''

    if st.button('Heart Disease Test Result'):
        data = [Age, Sex, CP, restbp, chol, FBS, restecg, thalach, exang, oldpeak, slope, CA, thal]
        heart_diagnosis = heart_disease_prediction(data)
        
        # Save to session state
        st.session_state.prediction_history.append({
            "Disease": "Heart Disease",
            "Input": data,
            "Result": heart_diagnosis
        })

    st.success(heart_diagnosis)


#Parkinson Disease Prediction page
elif(selected=='Parkinsons Disease Prediction'):
    #page title
    st.title('Parkinsons Disease Prediction')

    # Row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input("MDVP_Fo(Hz)", help="Average vocal fundamental frequency")
    with col2:
        fhi = st.text_input("MDVP_Fhi(Hz)", help="Maximum vocal fundamental frequency")
    with col3:
        flo = st.text_input("MDVP_Flo(Hz)", help="Minimum vocal fundamental frequency")
    with col4:
        Jitter_percent = st.text_input("MDVP:Jitter (%)", help="Variation in fundamental frequency")
    with col5:
        Jitter_abs = st.text_input("MDVP:Jitter (Abs)", help="Absolute jitter in Hz")

    # Row 2
    with col1:
        RAP = st.text_input("MDVP:RAP", help="Relative amplitude perturbation")
    with col2:
        PPQ = st.text_input("MDVP:PPQ", help="Period perturbation quotient")
    with col3:
        DDP = st.text_input("Jitter:DDP", help="Average difference of differences between cycles")
    with col4:
        Shimmer = st.text_input("Shimmer", help="Amplitude variation")
    with col5:
        Shimmer_dB = st.text_input("Shimmer(dB)", help="Shimmer in decibels")

    # Row 3
    with col1:
        APQ3 = st.text_input("Shimmer:APQ3", help="3-point amplitude perturbation quotient")
    with col2:
        APQ5 = st.text_input("Shimmer:APQ5", help="5-point amplitude perturbation quotient")
    with col3:
        APQ = st.text_input("MDVP:APQ", help="Average amplitude perturbation quotient")
    with col4:
        DDA = st.text_input("Shimmer:DDA", help="Diff. of amplitudes between cycles")
    with col5:
        NHR = st.text_input("NHR", help="Noise-to-harmonics ratio")

    # Row 4
    with col1:
        HNR = st.text_input("HNR", help="Harmonics-to-noise ratio")
    with col2:
        RPDE = st.text_input("RPDE", help="Recurrence period density entropy")
    with col3:
        DFA = st.text_input("DFA", help="Detrended fluctuation analysis")
    with col4:
        spread1 = st.text_input("Spread1", help="Nonlinear measure of signal")
    with col5:
        spread2 = st.text_input("Spread2", help="Another nonlinear measure")

    # Row 5
    with col1:
        D2 = st.text_input("D2", help="Correlation dimension")
    with col2:
        PPE = st.text_input("PPE", help="Pitch period entropy")

    parkinsons_diagnosis=''

    if st.button('Parkinsons Disease Test Result'):
        data = [fo, fhi, flo, Jitter_percent, Jitter_abs, RAP, PPQ, DDP, 
                Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, 
                RPDE, DFA, spread1, spread2, D2, PPE]
        parkinsons_diagnosis = parkinsons_prediction(data)
        
        # Save to session state
        st.session_state.prediction_history.append({
            "Disease": "Parkinson’s Disease",
            "Input": data,
            "Result": parkinsons_diagnosis
        })

    st.success(parkinsons_diagnosis)


#Dashboard
elif selected == 'Dashboard':
    st.title("Prediction Dashboard")

    if st.session_state.prediction_history:
        for i, entry in enumerate(st.session_state.prediction_history[::-1], 1):
            with st.expander(f"{i}. {entry['Disease']} Prediction"):
                st.write("**Input Data:**", entry["Input"])
                st.write("**Result:**", entry["Result"])
    else:
        st.info("No predictions have been made yet.")

