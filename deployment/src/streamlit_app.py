import streamlit as st
import eda as eda
import prediction as prediction

st.sidebar.title('Navigation')
navigation = st.sidebar.selectbox('Pages:', ('Exploratory Data Analysis', 'CVD Risk Level Prediction'))

if navigation == 'Exploratory Data Analysis':
    eda.run()
elif navigation == 'CVD Risk Level Prediction':
    prediction.run()

# hide navigation
st.markdown("""
<style>
[data-testid="stSidebarNav"]{
display: none;
}
</style>
"""
, unsafe_allow_html=True)
