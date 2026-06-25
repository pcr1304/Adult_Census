import streamlit as st
import pandas as pd
import pickle

# Inject custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
    }
    .title {
        font-size:40px;
        font-weight:600;
        color:#4B8BBE;
        text-align: center;
    }
    .subtitle {
        font-size:18px;
        text-align: center;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>💼  Income Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether an individual's income exceeds 50K/year 💰</div>", unsafe_allow_html=True)
st.write("")

# Load pre-trained model and preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("finalmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", min_value=0, max_value=120, value=30)
    workclass = st.selectbox("🏢 Workclass", [
        '?', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    fnlwgt = st.number_input("📊 Final Weight (fnlwgt)", value=100000)
    education = st.selectbox("🎓 Education", [
        'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate', 'Assoc-acdm',
        'Assoc-voc', '11th', '10th', '7th-8th', '5th-6th', '1st-4th', 'Preschool',
        '9th', '12th'
    ])
    education_num = st.number_input("📚 Education Num", min_value=1, max_value=20, value=9)
    marital_status = st.selectbox("💍 Marital Status", [
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed',
        'Married-spouse-absent'
    ])
    occupation = st.selectbox("💼 Occupation", [
        '?', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces'
    ])

with col2:
    relationship = st.selectbox("👪 Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
    ])
    race = st.selectbox("🌍 Race", [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])
    sex = st.selectbox("⚥ Sex", ['Male', 'Female'])
    capital_gain = st.number_input("📈 Capital Gain", value=0)
    capital_loss = st.number_input("📉 Capital Loss", value=0)
    hours_per_week = st.number_input("⏱ Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.selectbox("🏳️ Native Country", [
        'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England',
        'Cuba', 'Jamaica', 'South', 'China', 'Italy', 'Puerto-Rico', 'Vietnam', 'Honduras',
        'Japan', 'Iran', 'Ireland', '?'
    ])

# Submit button
if st.button("🔍 Predict Income"):
    input_dict = {
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education_num': [education_num],
        'marital_status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital_gain': [capital_gain],
        'capital_loss': [capital_loss],
        'hours_per_week': [hours_per_week],
        'native_country': [native_country]
    }

    input_df = pd.DataFrame(input_dict)

    try:
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"💡 Predicted Income: **{result}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
