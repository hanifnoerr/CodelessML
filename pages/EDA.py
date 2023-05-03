#importing library
import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
import matplotlib.pyplot as plt 
import seaborn as sns

st.cache_data

data = st.file_uploader('Upload your data here', type=['csv','xlsx', 'xls'])
if data is not None:
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

    #basic info of dataset
    st.write('uploaded',filename, 'with the shape of', df.shape)
    st.write('dataset sample')
    st.write(df.sample(5))

    #defining summary
    def dfSummary(data):
        summary = pd.DataFrame(data.dtypes,columns=['dtypes'])
        summary = summary.reset_index()
        summary['Column'] = summary['index']
        summary = summary[['Column','dtypes']]
        summary['non-null'] = data.notnull().sum().values
        summary['Missing'] = data.isnull().sum().values 
        summary['Missing (%)'] = data.isnull().sum().values * 100 / len(data) 
        summary['Uniques'] = data.nunique().values  
        return summary
    #create summary dataset
    dfsum = dfSummary(df)
    # Show Summary
    st.write("Summary of the dataset")
    st.dataframe(dfsum)
    
    st.write('Descriptive statistics')
    st.dataframe(df.describe())

    #create histogram for all numerical data
    if st.checkbox('Numerical data distribution'):
        fig = plt.figure(figsize = (15,15))
        ax = fig.gca()
        df.hist(ax=ax)
        st.pyplot(fig)

    #create plot for all categorical data, with unique value thresholds to mean
    if st.checkbox('Categorical data distribution'):
        # min_value = dfsum['Uniques'].min().astype(int).astype(object)  #.astype(object) is necessary to make int type instead of numpy.int64
        min_value = dfsum['Uniques'].min().astype(int).astype(object) 
        max_value = dfsum['Uniques'].max().astype(int).astype(object)
        current = dfsum['Uniques'].mean().round().astype(int).astype(object)
        uvt = st.slider('Unique Value Threshold', min_value, max_value, current)
        cat_data = dfsum[(dfsum['dtypes'] == 'object') & (dfsum['Uniques'] <= uvt)] 
        cat_df = df[cat_data.Column.values]
        
        fig = plt.figure(figsize = (15,15))
        for index in range(len(cat_data['Column'])):
            plt.subplot((len(cat_data['Column'])),4,index+1)
            sns.countplot(x=cat_df.iloc[:,index], data=cat_df.dropna())
            plt.xticks(rotation=90)
        st.pyplot(fig)
        
    if st.checkbox('Custom plot'):
        num_data = dfsum[dfsum['dtypes'] != 'object'] 
        num_df = df[num_data.Column.values]
        columns_list = num_df.columns.to_list()
        cat2_data = dfsum[dfsum['dtypes'] == 'object'] 
        cat2_df = df[cat2_data.Column.values]
        columns_cat_list = cat2_df.columns.to_list()
        chart_options = st.selectbox(
        'Select chart type',
        ('Boxplot', 'Scatterplot'))
        if chart_options == 'Boxplot':
            col1, col2 = st.columns(2)
            with col2:
                x_axis = st.selectbox(
                                'Select column for X axis',
                                (columns_cat_list))
            with col1:
                y_axis = st.selectbox(
                                'Select column for Y axis',
                                (columns_list))
                
            if st.button('Generate Chart'):
                fig, ax = plt.subplots()
                sns.boxplot(x=df[x_axis], y=df[y_axis])
                st.pyplot(fig)
            else:
                st.write('select chart type and column for the axis')
        
        if chart_options == 'Scatterplot':
            col1, col2 = st.columns(2)
            with col2:
                x_axis = st.selectbox(
                                'Select column for X axis',
                                (columns_list))
            with col1:
                y_axis = st.selectbox(
                                'Select column for Y axis',
                                (columns_list))
                
            if st.button('Generate Chart'):
                charts = plt.figure() 
                plt.scatter(df[x_axis], df[y_axis])
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                st.pyplot(charts)
            else:
                st.write('select chart type and column for the axis')

    #correlation plot
    if st.checkbox('Correlation plot'):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(),  ax=ax, vmin=-1, vmax=1, annot=True)
        st.pyplot(fig)

