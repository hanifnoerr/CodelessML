#importing library
import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
#load model's libraries
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

#another libs
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from zipfile import ZipFile
from io import BytesIO
import joblib


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
            
    if 'selected_target_value' not in st.session_state:
        st.session_state['selected_target_value'] = None
        
    with st.form('modelling'):
        st.write('Default data processing will be used for the following:')
        #summary
        def dfSummary(data):
            summary = pd.DataFrame(data.dtypes,columns=['dtypes'])
            summary = summary.reset_index()
            summary['Column'] = summary['index']
            summary = summary[['Column']]
            summary['Missing'] = data.isnull().sum().values * 100 / len(data) 
            summary['Uniques'] = data.nunique().values * 100 / len(data)
            return summary
        #create summary dataset
        dfsum = dfSummary(df)
        threshold = 40
        dropped = dfsum[(dfsum['Missing'] >= threshold) | (dfsum['Uniques'] == 100)] #drop columns with equal or more than 40% of missing values & drop columns that is 100% unique
        list_dropped = dropped.Column.to_list()
        d = ''
        for i in list_dropped:
            d += "- " + i + "\n"
        st.write('Dropped:')
        if not list_dropped:
            st.write("- No column will be dropped")
        st.markdown(d)
        keep = dfsum[(dfsum['Missing'] < threshold) & (dfsum['Missing'] > 0) ] 
        list_keep = keep.Column.to_list()
        st.write('Columns which missing values will be replaced with mean (int) and mode (str):')
        k = ''
        for i in list_keep:
            k += "- " + i + "\n"
        if not list_keep:
            st.write("- No column with missing values")
        st.markdown(k)
        #determine the target value
        new_df = df.drop(list_dropped, axis=1)
        pilihan = new_df.columns.to_list()
        empty_val = ''
        pilihan.insert(0, empty_val)
        col1, col2 = st.columns(2)
        with col1:
            target_values = st.selectbox('select target column',
                            (pilihan))
        with col2:
            splitRatio = st.slider('Data Splitting Ratio', 10, 90, 70)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['selected_target_value'] = target_values
            st.session_state['selected_target_ratio'] = splitRatio
            
    if target_values != '':
        ################
        #inputing missing values
        int_columns = new_df.select_dtypes(np.number).columns
        obj_columns = new_df.select_dtypes(exclude=np.number).columns
        new_df[int_columns] = new_df[int_columns].apply(lambda x: x.fillna(x.mean()))
        new_df[obj_columns] = new_df[obj_columns].apply(lambda x: x.fillna(x.mode()[0]))
        #separare non numerical columns
        obj_columns = obj_columns.to_list()
        #this is necessary because Ordinal encoder fails when there are spaces in the column
        for stripSpaces in obj_columns:
            new_df[stripSpaces] = new_df[stripSpaces].str.replace(' ', '')
        #encoders
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        objToInt = encoder.fit_transform(new_df[obj_columns])
        new_df[obj_columns] = pd.DataFrame(objToInt, columns=obj_columns)
        #split train & test
        target = target_values
        y = new_df[target]
        X = new_df.drop(target, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-splitRatio/100, random_state=7)
        st.text(f'Using {splitRatio} % of the data as data train, with the splitting result:')
        st.write('train shape:',X_train.shape , 'test shape:', X_test.shape)
        #modelling classficiation
        st.write('training...')
        st.write('Done!')
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test) 

        r2_lr = r2_score(y_test, y_pred_lr)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
        
        #Support Vector Regression
        svmr = SVR()
        svmr.fit(X_train, y_train)
        y_pred_svmr = svmr.predict(X_test) 

        r2_svmr = r2_score(y_test, y_pred_svmr)
        mse_svmr = mean_squared_error(y_test, y_pred_svmr)
        rmse_svmr = mean_squared_error(y_test, y_pred_svmr, squared=False)
        mae_svmr = mean_absolute_error(y_test, y_pred_svmr)
        mape_svmr = mean_absolute_percentage_error(y_test, y_pred_svmr)

        # K-Nearest Neighbor
        knn = KNeighborsRegressor()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test) 

        r2_knn = r2_score(y_test, y_pred_knn)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        rmse_knn = mean_squared_error(y_test, y_pred_knn, squared=False)
        mae_knn = mean_absolute_error(y_test, y_pred_knn)
        mape_knn = mean_absolute_percentage_error(y_test, y_pred_knn)
        
        # ElasticNet
        eln = ElasticNet(random_state=77)
        eln.fit(X_train, y_train)
        y_pred_eln = eln.predict(X_test) 

        r2_eln = r2_score(y_test, y_pred_eln)
        mse_eln = mean_squared_error(y_test, y_pred_eln)
        rmse_eln = mean_squared_error(y_test, y_pred_eln, squared=False)
        mae_eln = mean_absolute_error(y_test, y_pred_eln)
        mape_eln = mean_absolute_percentage_error(y_test, y_pred_eln)

        # Passive Aggressive Regressor
        par = PassiveAggressiveRegressor(random_state=77)
        par.fit(X_train, y_train)
        y_pred_par = par.predict(X_test) 

        r2_par = r2_score(y_test, y_pred_par)
        mse_par = mean_squared_error(y_test, y_pred_par)
        rmse_par = mean_squared_error(y_test, y_pred_par, squared=False)
        mae_par = mean_absolute_error(y_test, y_pred_par)
        mape_par = mean_absolute_percentage_error(y_test, y_pred_par)
        
        # Random Forest
        rf = RandomForestRegressor(random_state=77)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test) 

        r2_rf = r2_score(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)


        # Gradient Boosting Regressor
        gbr = GradientBoostingRegressor(random_state=77)
        gbr.fit(X_train, y_train)
        y_pred_gbr = gbr.predict(X_test) 

        r2_gbr = r2_score(y_test, y_pred_gbr)
        mse_gbr = mean_squared_error(y_test, y_pred_gbr)
        rmse_gbr = mean_squared_error(y_test, y_pred_gbr, squared=False)
        mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
        mape_gbr = mean_absolute_percentage_error(y_test, y_pred_gbr)
        
        # LGBM
        lgbm = LGBMRegressor(random_state=77)
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test) 

        r2_lgbm = r2_score(y_test, y_pred_lgbm)
        mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
        rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
        mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
        mape_lgbm = mean_absolute_percentage_error(y_test, y_pred_lgbm)
        
        # XGBoost
        xgboost = xgb.XGBRegressor(random_state=77)
        xgboost.fit(X_train, y_train)
        y_pred_xgboost = xgboost.predict(X_test) 

        r2_xgboost = r2_score(y_test, y_pred_xgboost)
        mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
        rmse_xgboost = mean_squared_error(y_test, y_pred_xgboost, squared=False)
        mae_xgboost = mean_absolute_error(y_test, y_pred_xgboost)
        mape_xgboost = mean_absolute_percentage_error(y_test, y_pred_xgboost)

        #compile the result to dataframe 
        models = pd.DataFrame({
            'Model': ['Linear Regression', 'Support Vector Regression', 'K-Nearest Neighbor', 'Elastic Net','Passive Aggressive Regressor', 'Random Forest','Gradient Boosting', 'LightGBM', 'XGBoost'],
            'R2': [r2_lr, r2_svmr, r2_knn, r2_eln, r2_par, r2_rf, r2_gbr,r2_lgbm, r2_xgboost],
            'MSE': [mse_lr, mse_svmr, mse_knn, mse_eln, mse_par, mse_rf, mse_gbr, mse_lgbm, mse_xgboost],
            'RMSE': [rmse_lr, rmse_svmr, rmse_knn, rmse_eln, rmse_par, rmse_rf, rmse_gbr, rmse_lgbm, rmse_xgboost],
            'MAE': [mae_lr, mae_svmr, mae_knn, mae_eln, mae_par, mae_rf, mae_gbr, mae_lgbm, mae_xgboost],
            'MAPE': [mape_lr, mape_svmr, mape_knn, mape_eln, mape_par, mape_rf, mape_gbr, mape_lgbm, mape_xgboost]
                            })
        st.write('sample data used as train')
        st.dataframe(X.head(3))
        
        st.write('Result & Model Comparison')
        st.dataframe(models.sort_values(by='RMSE', ascending=True).style.format())
    
    
        ########################################################################
        model_alias = {
                    'lr':'Linear Regression', 
                    'svmr':'Support Vector Regression', 
                    'knn':'K-Nearest Neighbor', 
                    'eln':'Elastic Net',
                    'par':'Passive Aggressive Regressor', 
                    'rf':'Random Forest',
                    'gbr':'Gradient Boosting', 
                    'lgbm':'LightGBM', 
                    'xgboost':'XGBoost'}

        def alias_func(selected_model):
            return model_alias[selected_model]

                
        selected_model = st.selectbox("Select model to download", options=list(model_alias.keys()), format_func=alias_func)
        joblib.dump(encoder, 'encoder.pkl')
        #due some bug, we throw all models
        joblib.dump(lr, 'lr.joblib')

        if selected_model == 'lr':
            joblib.dump(lr, 'model.joblib')
        if selected_model == 'svm':
            joblib.dump(svmr, 'model.joblib')
        if selected_model == 'knn':
            joblib.dump(knn, 'model.joblib')
        if selected_model == 'mnb':
            joblib.dump(eln, 'model.joblib')
        if selected_model == 'dt':
            joblib.dump(par, 'model.joblib')
        if selected_model == 'rf':
            joblib.dump(rf, 'model.joblib')
        if selected_model == 'gbc':
            joblib.dump(gbr, 'model.joblib')
        if selected_model == 'lgbm':
            joblib.dump(lgbm, 'model.joblib')
        if selected_model == 'xgboost':
            joblib.dump(xgboost, 'model.joblib')
        #create csv file that have been encoded
        new_df.to_csv('encoded_data.csv', sep=';')
        


        temp = BytesIO()
        
        with ZipFile(temp, "x") as csv_zip:
            csv_zip.write("encoder.pkl")
            csv_zip.write("model.joblib")
            csv_zip.write("encoded_data.csv")

        st.download_button(
            label=f"Download {model_alias.get(selected_model)} Model",
            data=temp.getvalue(),
            file_name=f"{model_alias.get(selected_model)}.zip",
            mime="application/zip",
        )
