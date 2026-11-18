import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

MODEL_PATH = os.path.join(BASE_DIR, 'best_gradboost_model.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

with open(MODEL_PATH, 'rb') as model_file:
    best_gradboost_model = pickle.load(model_file)

with open(LABEL_ENCODER_PATH, 'rb') as label_encoder_file:
    le = pickle.load(label_encoder_file)

st.info('This app is built with a machine learning model!')

def run():
    st.title('Estimate Your Risk Level of CVD')
    st.subheader('Please fill in the screening form below:')

    with st.form(key='Form Parameters'):
        Sex=st.selectbox('Sex: Female or Male', ['F', 'M'], index=1)
        Age=st.number_input('Age', min_value=1, max_value=150, value=30)
        Weight= st.number_input('Weight (kg)', min_value=5, max_value=500,value=55)
        Height_m=st.number_input('Height (m)', value=1.63)
        BMI=st.number_input('BMI',value=0)
        AbdominalCircumference= st.number_input('Abdominal Circumference (cm)', value=0)

        st.markdown('---')
        BloodPressure=st.text_input('Blood Pressure (mmHg)', value='110/80')
        TotalCholesterol= st.number_input('Total Cholesterol(mg/dL)', value=0)
        HDL= st.number_input('HDL (mg/dL)', value=0)
        FastingBloodSugar=st.number_input('Fasting Blood Sugar', value=0)

        st.markdown('---')
        SmokingStatus = st.selectbox('Smoking Status: Yes or No', ['Y', 'N'], index=1)
        DiabetesStatus = st.selectbox('Diabetes Status: Yes or No', ['Y', 'N'], index=1)
        PhysicalActivityLevel = st.selectbox('Physical Activity Level', ['Low', 'Moderate', 'High'], index=2)
        FamilyHistoryofCVD = st.selectbox('Family History of CVD: Yes or No', ['Y', 'N'], index=1)

        st.markdown('---')
        Height_cm = st.number_input('Height (cm)', value=163)
        WaistToHeightRatio = st.number_input('Waist-to-Height Ratio', value=0)
        SystolicBP = st.number_input('Systolic BP', value=110)
        DiastolicBP = st.number_input('Diastolic BP', value=90)
        BloodPressureCategory=st.selectbox('Blood Pressure category', ['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2'])
        EstimatedLDL= st.number_input('Estimated LDL', value=100)
        #CVDRiskScore tidak dimasukkan form, karena tidak masuk di feature selection
        # selain itu, di medical check up standard, tidak ada CVD risk.
        # CVDRiskScore= st.number_input('CVD Risk Score', value=10.5)

        submit = st.form_submit_button('Predict')

    # define data inference
    data_inf={
        'Sex': Sex,
        'Age': Age,
        'Weight':  Weight,
        'Height_m': Height_m,
        'BMI': BMI,
        'Abdominal Circumference': AbdominalCircumference,
        'Blood Pressure (mmHg)': BloodPressure,
        'Total Cholesterol': TotalCholesterol,
        'HDL': HDL,
        'Fasting Blood Sugar': FastingBloodSugar,
        'Smoking Status': SmokingStatus,
        'Diabetes Status': DiabetesStatus,
        'Physical Activity Level': PhysicalActivityLevel,
        'Family History of CVD': FamilyHistoryofCVD,
        'Height_cm': Height_cm,
        'Waist-to-Height Ratio': WaistToHeightRatio,
        'Systolic BP': SystolicBP,
        'Diastolic BP': DiastolicBP,
        'Blood Pressure Category': BloodPressureCategory,
        'Estimated LDL': EstimatedLDL,
        'CVD Risk Score': np.nan
    }

    
    # input data inference into dataframe
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submit:
        #predict
        y_pred_original = best_gradboost_model.predict(data_inf)
        y_pred_inf = le.inverse_transform(y_pred_original)

        st.markdown('## Prediction Result')
        st.markdown(f'### {y_pred_inf}')

if __name__=='__main__':
    run()
