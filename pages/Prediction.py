#importing library
import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
import glob
from sklearn.preprocessing import OrdinalEncoder
#load model's libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

#another libs
from zipfile import ZipFile
from io import BytesIO
import joblib

st.cache_data

data = st.file_uploader('Upload your data here, the file name must be "test" (ex: test.csv)', type=['csv','xlsx', 'xls'])
encoder = st.file_uploader('Please upload encoder.pkl', type=['pkl'])
model = st.file_uploader('Please upload model.joblib', type=['joblib'])
if data and encoder and model is not None:
    #check the type of file uploaded and get the extension of the file
    filename=data.name
    x = filename.rsplit('.', 1)

    #if uploaded file is CSV
    if x[1] == 'csv':
        # because there are many posibilites for csv delimiter, using string io then combine it with csv sniffer to get the delimiter format
        stringio = StringIO(data.getvalue().decode("utf-8"))
        string_data = stringio.read()
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(string_data)
        delimiter = dialect.delimiter
        df = pd.read_csv(data, sep=dialect.delimiter)
    #if uploaded file is excel file
    else:
        df = pd.read_excel(data) #catatan perlu openpyxl sama xlrd (buat xls)

    #load model & encoder
    modelnya = joblib.load('model.joblib')
    encoder_prod = joblib.load('encoder.pkl')
    columns_list = df.columns.to_list()
    options = st.multiselect(
    'select columns to ignore, column names should match those that were used during train in modelling',columns_list)
    final_data = df[df.columns[~df.columns.isin(options)]]
    st.write('data sample:')
    st.write(final_data.sample(3))
    obj_columns = final_data.select_dtypes(exclude=np.number).columns
    for stripSpaces in obj_columns:
        final_data[stripSpaces] = final_data[stripSpaces].str.replace(' ', '')
    # Apply encoder_prod to unknown data
    final_data[obj_columns] = encoder_prod.transform(final_data[obj_columns])
    final_data.fillna(-1, inplace=True)
    prediction = modelnya.predict(final_data) 
    #set the output as a dataframe 
    output = pd.DataFrame({'prediction': prediction })

    final_output = pd.concat([df, output], axis=1)
    st.write("Predicting...")
    st.write("Done!")
    st.write("sample output:")
    st.write(final_output.head(5))
    
    def convert_df(final_output):
        return final_output.to_csv(index=False).encode('utf-8')


    csv = convert_df(final_output)

    st.download_button(
    "Download Output File",
    csv,
    "prediction.csv",
    "text/csv",
    key='download-csv'
    )
    
    