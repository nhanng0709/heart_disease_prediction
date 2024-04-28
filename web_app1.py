import pandas as pd
heart_data = pd.read_csv('health_disease_dataset.csv')

from imblearn.over_sampling import SMOTE
columns_to_exclude = ['Fruits', 'Veggies', 'Education', 'Income', 'MentHlth']
x = heart_data.drop(['HeartDiseaseorAttack'] + columns_to_exclude, axis=1)
y=heart_data[['HeartDiseaseorAttack']]
smote = SMOTE(random_state=42)
x_balanced, y_balanced = smote.fit_resample(x, y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, test_size=0.3, random_state=42)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
Rforest = RandomForestClassifier(random_state=11)
Rforest.fit(x_train, y_train)
predictions_rf = Rforest.predict(x_test)


import streamlit as st
# Define the prediction function
def predict_heart_disease(high_bp, high_chol, chol_check, bmi, smoker, stroke, diabetes, phys_activity,
                          heavy_alcohol_consump, any_healthcare, no_doc_cost, general_health,
                          physical_health, diff_walk, sex, age):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'HighBP': [high_bp],
        'HighChol': [high_chol],
        'CholCheck': [chol_check],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Stroke': [stroke],
        'Diabetes': [diabetes],
        'PhysActivity': [phys_activity],
        'HvyAlcoholConsump': [heavy_alcohol_consump],
        'AnyHealthcare': [any_healthcare],
        'NoDocbcCost': [no_doc_cost],
        'GenHlth': [general_health],
        'PhysHlth': [physical_health],
        'DiffWalk': [diff_walk],
        'Sex': [sex],
        'Age': [age]
    })
    
    # Make prediction using the random forest model
    prediction = Rforest.predict(input_data)
    prediction_prob = Rforest.predict_proba(input_data)[:, 1] *100
    
    return prediction, prediction_prob

# Define the Streamlit app
def main():
    # Add a title to the app
    st.title("Heart Disease Prediction App")

    # Add input widgets for user input
    high_bp = st.radio("Do you have High Blood Pressure? (Yes = 1, No =0):", options=[0, 1])
    high_chol = st.radio("Do you have High Cholesterol? (Yes = 1, No =0):", options=[0, 1])
    chol_check = st.radio("Have you had Cholesterol Check within the past 5 years? (Yes = 1, No =0):", options=[0, 1])
    bmi = st.slider("BMI:", min_value=10, max_value=50, value=25, step=1)
    smoker = st.radio("Hаvе уоu ѕmоkеd аt least 100 сіgаrеttеѕ in уоur еntіrе lіfе? (Yes = 1, No =0) :", options=[0, 1])
    stroke = st.radio("Have уоu ever hаd a ѕtrоkе? (Yes = 1, No =0):", options=[0, 1])
    diabetes = st.selectbox("Diabetes:0 is no dіаbеtеѕ, 1 іѕ рrе-dіаbеtеѕ, and 2 іѕ diabetes", options=[0, 1, 2])
    phys_activity = st.radio("Have you done physical асtіvіtу or exercise during the past 30 days? (Yes = 1, No =0):", options=[0, 1])
    heavy_alcohol_consump = st.radio("Heavy Alcohol Consumption (Yes = 1, No =0):", options=[0, 1])
    any_healthcare = st.radio("Dо уоu hаvе аnу kind of health саrе coverage, including hеаlth іnѕurаnсе, prepaid plans ѕuсh as HMOѕ, оr government рlаnѕ ѕuсh as Mеdісаrе, оr Indian Hеаlth Sеrvісе? (Yes = 1, No =0) :", options=[0, 1])
    no_doc_cost = st.radio("Wаѕ there a time іn thе раѕt 12 mоnthѕ whеn уоu needed tо ѕее a dосtоr but could nоt because оf cost? (Yes = 1, No =0):", options=[0, 1])
    general_health = st.slider("Would уоu say thаt іn general, уоur hеаlth іѕ: ( from 1: extremely bad to 5 : extremely good):", min_value=1, max_value=5, value=3, step=1)
    physical_health = st.slider("Which іnсludеѕ рhуѕісаl illness аnd іnjurу, fоr hоw mаnу dауѕ during the раѕt 30 dауѕ wаѕ уоur physical hеаlth nоt good? (From 1 to 30 days) :", min_value=0, max_value=30, value=0, step=1)
    diff_walk = st.radio("Dо уоu hаvе ѕеrіоuѕ difficulty wаlkіng оr сlіmbіng ѕtаіrѕ? (Yes = 1, No =0):", options=[0, 1])
    sex = st.radio("Sex (Male = 1, No =0):" , options=[0, 1])
    age = st.slider("Age:", min_value=20, max_value=90, value=50, step=1)

    # Add a button to trigger prediction
    if st.button("Predict"):
        prediction, prediction_prob = predict_heart_disease(high_bp, high_chol, chol_check, bmi, smoker, stroke,
                                                            diabetes, phys_activity, heavy_alcohol_consump,
                                                            any_healthcare, no_doc_cost, general_health,
                                                            physical_health, diff_walk, sex, age)
        st.write("Prediction Probability of getting Heart Disease:", prediction_prob)

# Run the app
if __name__ == "__main__":
    main()