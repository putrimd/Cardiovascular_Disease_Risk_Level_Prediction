import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'P1M2_putri_dataset_cardiovascular.csv')


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
    
st.set_page_config(
    page_title='Cardiovascular Risk Level EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    st.title('Cardiovascular Risk Level Data Analysis')

    st.subheader('Created by: Putri')

    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYW3jGMnCeEkIdb95-K7_DSCg8p9x9oQDv5rw-6dvD2Q&s=10',
             caption='Sayangi jantung Anda')

    st.markdown('Cardiovascular Disease (CVD) adalah penyakit yang terjadi karena adanya gangguan jantung dan pembuluh darah. Penyakit Jantung dan stroke merupakan contoh penyakit Cardiovascular yang terkenal.')
    st.markdown('Berdasarkan WHO, estimasi 19.8 juta orang meninggal di tahun 2022, dikarenakan CVD, dengan 85% diantaranya serangan jantung dan stroke.')
    st.markdown('Di Indonesia, CVD menyebabkan sekitar 651.481 kematian setiap tahunnya, yang dikarenakan pola hidup tidak sehat, seperti terlalu banyak makan makanan berlemak, tidak rutin olahraga, kebiasaan merokok, dan kebiasaan konsumsi alkohol (Alodokter, 2026).')
    st.markdown('Sebagai bentuk mitigasi penangan CVD awal, maka dibutuhkan data screening yang dapat memprediksi seseorang memiliki resiko penyakit Cardiovascular.')
    st.markdown('Project ini dibuat dengan menggunakan algoritma GradientBoosting.')
    st.markdown('-'*40)
    df = load_data()
    with st.expander('View the Data'):
        st.write('**Raw Data of 1529 Respondents Jamalpur Medical College Hospital, Bangladesh**')
        st.dataframe(df)

        st.write('**X Features**')
        st.dataframe(df.drop('CVD Risk Level', axis=1))

        st.write('**y target**')
        st.dataframe(df['CVD Risk Level'])
        

    # first viz
    st.header('CVD Risk Level of Respondents')
    fig=plt.figure(figsize=(16,8))
    sns.countplot(data=df, x='CVD Risk Level')
    st.pyplot(fig)

    with st.expander('Insight'):
        st.write(' Dari 1529 Responden diketahui:728 orang berpotensi memiliki CVD Risk Level **High**, 581 orang berpotensi memiliki CVD Risk Level **Intermediary**, 220 orang berpotensi memiliki CVD Risk Level **Low**')

    # second viz
    st.header('Data Histogram')
    optionlist= st.selectbox('Choose Data:',['Age', 'Height (m)', 'Weight (kg)'])
    fig=plt.figure(figsize=(16,8))
    sns.histplot(df[optionlist], kde=True, bins=15)
    st.pyplot(fig)

    # third viz
    st.header('Smoking Status vs CVD Risk Level')
    fig=plt.figure(figsize=(16,8))
    sns.countplot(data=df, x='Smoking Status', hue='CVD Risk Level')
    plt.legend()
    st.pyplot(fig)

    with st.expander('Insight'):
            st.write('Dari 1529 Responden diketahui: 455 perokok aktif berpotensi memiliki CVD Risk Level **High**, 231 perokok aktif berpotensi memiliki CVD Risk Level **Intermediary**, 117 perokok pasif berpotensi memiliki CVD Risk Level **Low**, Secara keseluruhan Perokok aktif cenderung memiliki CVD Risk Level High daripada perokok pasif.')
    
    # fourth viz
    st.header('Diabetes Status vs CVD Risk Factor Relationship')
    fig=plt.figure(figsize=(16,8))
    sns.countplot(data=df, x='Diabetes Status', hue='CVD Risk Level')
    plt.legend()
    st.pyplot(fig)

    with st.expander('Insight'):
         st.write(' Responden yang memiliki riwayat diabetes cenderung berpotensi memiliki CVD Risk Level High.')

    

if __name__ == '__main__':
    run()
    
