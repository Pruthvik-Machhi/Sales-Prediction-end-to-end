import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pickle

with open('model3a.pkl', 'rb') as model_file2:
    xgb = pickle.load(model_file2)

with open('minmax.pkl', 'rb') as scaler_file2:
    scaler2 = pickle.load(scaler_file2)


html_attribution = """
    <div style="background-color:#28a745;padding:20px;margin-bottom:20px">
    <p style="color:white;text-align:center;font-size:22px;">Developed by Pruthvik Machhi</p>
    </div>
    """
st.markdown(html_attribution, unsafe_allow_html=True)

 
html_temp_subtitle = """
    <div style="background-color:#007bff;padding:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;">Sales Prediction</h2>
    </div>
    """
st.markdown(html_temp_subtitle, unsafe_allow_html=True)

def user_input_features():
    i1 = st.number_input('TV')
    i2 = st.number_input('Radio')
    i3 = st.number_input('Newspaper')
    data = {'TV': i1, 'Radio': i2, 'Newspaper': i3}
    features = pd.DataFrame(data, index=[0])
    return features

st.subheader('Enter Input ')
df = user_input_features()

st.subheader('Input parameters')
st.write(df)

expected_features = ['TV', 'Radio', 'Newspaper']
df = df[expected_features]

if st.button('Predict'):
    scaled_features = scaler2.transform(df)
    prediction = xgb.predict(scaled_features)
    predicted_sales = prediction[0]
    st.subheader('Prediction')
    st.write(f"The predicted sales amount is: **{predicted_sales}**")
