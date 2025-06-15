import streamlit as st
import joblib
import pandas as pd

model = joblib.load('house_price_model.pkl')

st.title('House Price prediction')
st.write('Enter house details to predict the sale price.')

lot_area = st.number_input('Lot Area (sq ft)', min_value=0, value=10000)
bedrooms = st.number_input('Number of Bedrooms', min_value=0, value=3)
bathrooms = st.number_input('Number of Full Bathrooms', min_value=0, value=2)

if st.button('Predict price'):
    input_data = pd.DataFrame([[lot_area, bedrooms, bathrooms]], columns = ['LotArea', 'BedroomAbvGr', 'FullBath'])
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted House Price: ${prediction:,.2f}')

st.write('Note: Model trained on kaggle House Prices dataset.')