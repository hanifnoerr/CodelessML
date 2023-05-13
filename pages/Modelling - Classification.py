#importing library
import streamlit as st
import pandas as pd
import numpy as np
import csv
from io import StringIO
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        st.write('the following task will be executed:')
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
        st.write('Replaced the missing value with mean (int) and mode (str):')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-splitRatio/100, random_state=77, stratify=y)
        st.text(f'Using {splitRatio} % of the data as data train, with the splitting result:')
        st.write('train shape:',X_train.shape , 'test shape:', X_test.shape)
        st.write('sample data used as train')
        st.dataframe(X.head(3))
        #modelling classficiation
        st.write('training...')
        #LR
        lr = LogisticRegression(random_state=77, max_iter=10000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test) 

        recall_lr = recall_score(y_test, y_pred_lr, average='macro')
        acc_lr = accuracy_score(y_test, y_pred_lr)
        prec_lr = precision_score(y_test, y_pred_lr, average='macro')
        f1_lr = f1_score(y_test, y_pred_lr, average='macro')
        
        #Support Vector Machine Classifier (LinearSVC)
        svm = LinearSVC(random_state=77, dual=False)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test) 

        recall_svm = recall_score(y_test, y_pred_svm, average='macro')
        acc_svm = accuracy_score(y_test, y_pred_svm)
        prec_svm = precision_score(y_test, y_pred_svm, average='macro')
        f1_svm = f1_score(y_test, y_pred_svm, average='macro')

        #KNN (K-Nearest Neighbor)
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test) 

        recall_knn = recall_score(y_test, y_pred_knn, average='macro')
        acc_knn = accuracy_score(y_test, y_pred_knn)
        prec_knn = precision_score(y_test, y_pred_knn, average='macro')
        f1_knn = f1_score(y_test, y_pred_knn, average='macro')

        #NB 
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        y_pred_mnb = mnb.predict(X_test) 

        recall_mnb = recall_score(y_test, y_pred_mnb, average='macro')
        acc_mnb = accuracy_score(y_test, y_pred_mnb)
        prec_mnb = precision_score(y_test, y_pred_mnb, average='macro')
        f1_mnb = f1_score(y_test, y_pred_mnb, average='macro')

        ##DT
        dt = DecisionTreeClassifier(random_state=77)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test) 
        
        recall_dt = recall_score(y_test, y_pred_dt, average='macro')
        acc_dt = accuracy_score(y_test, y_pred_dt)
        prec_dt = precision_score(y_test, y_pred_dt, average='macro')
        f1_dt = f1_score(y_test, y_pred_dt, average='macro')
        
        #RF 
        rf = RandomForestClassifier(random_state=77)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test) 

        recall_rf = recall_score(y_test, y_pred_rf, average='macro')
        acc_rf = accuracy_score(y_test, y_pred_rf)
        prec_rf = precision_score(y_test, y_pred_rf, average='macro')
        f1_rf = f1_score(y_test, y_pred_rf, average='macro')

        #GBC
        gbc = GradientBoostingClassifier(random_state=77)
        gbc.fit(X_train, y_train)
        y_pred_gbc = gbc.predict(X_test) 

        recall_gbc = recall_score(y_test, y_pred_gbc, average='macro')
        acc_gbc = accuracy_score(y_test, y_pred_gbc)
        prec_gbc = precision_score(y_test, y_pred_gbc, average='macro')
        f1_gbc = f1_score(y_test, y_pred_gbc, average='macro')

        #lgbm
        lgbm = LGBMClassifier(random_state=77)
        lgbm.fit(X_train, y_train)
        y_pred_lgbm = lgbm.predict(X_test) 

        recall_lgbm = recall_score(y_test, y_pred_lgbm, average='macro')
        acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
        prec_lgbm = precision_score(y_test, y_pred_lgbm, average='macro')
        f1_lgbm = f1_score(y_test, y_pred_lgbm, average='macro')
        
        #xgb
        xgboost = xgb.XGBClassifier(random_state=77)
        xgboost.fit(X_train, y_train)
        y_pred_xgboost = xgboost.predict(X_test) 

        recall_xgboost = recall_score(y_test, y_pred_xgboost, average='macro')
        acc_xgboost = accuracy_score(y_test, y_pred_xgboost)
        prec_xgboost = precision_score(y_test, y_pred_xgboost, average='macro')
        f1_xgboost = f1_score(y_test, y_pred_xgboost, average='macro')


        #compile the result to dataframe 
        models = pd.DataFrame({
            'Model': ['Logistic Regression', 'Linear SVC', 'K-Nearest Neighbor', 'Multinomial Naive Bayes','Decision Tree', 'Random Forest','Gradient Boosting', 'LightGBM', 'XGBoost'],
            'Recall': [recall_lr, recall_svm, recall_knn, recall_mnb, recall_dt, recall_rf, recall_gbc,recall_lgbm, recall_xgboost],
            'Accuracy': [acc_lr, acc_svm, acc_knn, acc_mnb, acc_dt, acc_rf, acc_gbc, acc_lgbm, acc_xgboost],
            'Precission': [prec_lr, prec_svm, prec_knn, prec_mnb, prec_dt, prec_rf, prec_gbc, prec_lgbm, prec_xgboost],
            'F1 Score': [f1_lr, f1_svm, f1_knn, f1_mnb, f1_dt, f1_rf, f1_gbc, f1_lgbm, f1_xgboost],
                            })
        st.write('Done!')
        
        st.write('Result & Model Comparison')
        st.dataframe(models.sort_values(by='F1 Score', ascending=False).style.format())
    
    
        ########################################################################
        model_alias = {
                    'lr':'Logistic Regression', 
                    'svm':'Linear SVC', 
                    'knn':'K-Nearest Neighbor', 
                    'mnb':'Multinomial Naive Bayes',
                    'dt':'Decision Tree', 
                    'rf':'Random Forest',
                    'gbc':'Gradient Boosting', 
                    'lgbm':'LightGBM', 
                    'xgboost':'XGBoost'}

        def alias_func(selected_model):
            return model_alias[selected_model]

                
        selected_model = st.selectbox("Select model to download", options=list(model_alias.keys()), format_func=alias_func)
        joblib.dump(encoder, 'encoder.pkl')

        if selected_model == 'lr':
            joblib.dump(lr, 'model.joblib')
        if selected_model == 'svm':
            joblib.dump(svm, 'model.joblib')
        if selected_model == 'knn':
            joblib.dump(knn, 'model.joblib')
        if selected_model == 'mnb':
            joblib.dump(mnb, 'model.joblib')
        if selected_model == 'dt':
            joblib.dump(dt, 'model.joblib')
        if selected_model == 'rf':
            joblib.dump(rf, 'model.joblib')
        if selected_model == 'gbc':
            joblib.dump(gbc, 'model.joblib')
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
