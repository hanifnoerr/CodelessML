import streamlit as st

st.image('CodelessML.png')
st.caption('A web-based application for anyone who wants to learn and create machine learning model without coding.')

st.subheader('What is CodelessML')
st.markdown(" **CodelessML** presents a general workflow for easing the process of creating a machine learning model and using the ML model for prediction. This application aims to help non-experts users experience the journey of how a machine learning model is built without requiring them to code. The application has three main menus: EDA, Modelling, and Prediction.")
st.markdown(" - The _EDA_ menu is used to explore and performs automatic visualisation of any dataset without writing a single code. Just upload your data in a compatible file format (csv or Excel).")
st.markdown(" - The _Modelling_ menu is used to create machine learning models; you need to select the type of task you want to perform: Classification or Regression. This menu also performs data processing and perform an evaluation of all 9 ML algorithms available in this application. For now, only default data processing, such as dropping columns that have equal or more than 40% of missing values, dropping columns that are 100% unique, inputting missing values with their mean if the data type is number or mode if the data type is a string is available. These steps are necessary because most machine learning algorithms can't deal with missing values and are automatically executed when you click submit button on the _Modelling_ menu.")
st.markdown(" - The _Prediction_ menu is used to predict new data for which you do not know the target using a trained model.")
st.markdown(" **CodelessML** is better for non-expert learners because:")
st.markdown(" - **CodelessML** can be access from anywhere with an internet connection, which means you can access the application from your home, office, or any other location of they desire.")
st.markdown(" - No installation required: you don't need to install any software on your computer, phone or other device to use **CodelessML**, which makes it more convenient and accessible. Just use your preferred browser to access **CodelessML**")
st.markdown(" - **CodelessML** offer an interactive learning experience where you can see the impact of their choices in real-time and explore different scenarios without writing a single line of code.")
st.subheader('How to use')
st.markdown(" 1. Explore and understand your data using the _EDA_ menu (e.g., descriptive statistics, data visualisation, correlation matrix, etc.)")
st.markdown(" 2. Go to the _Modelling_ menu. _[Modelling  - Classification]_  for the classification task, and _[Modelling – Regression]_  for the regression task. Choose the menus that best suit your problem. Then select the target variable and determine the ratio of data that will be used for training and testing your model. Click submit, and the application will automatically perform data processing and training ML models. ")
st.markdown(" 3. Determine which model you want to save and Click “Select model to download”. You will get model.joblib and encoder.pkl as well as the original data that have been encoded with label encoder inside a zipped folder.")
st.markdown(" 4. Use the model.joblib and encoder.pkl to predict your new data in the “Prediction Menu” and save the result by clicking the “Downlaod Output File” button.")
