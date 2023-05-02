# CodelessML

A web-based application to making Machine Learning tasks easier for anyone who wants to learn and create machine learning model without coding.

# About

CodelessML presents a general workflow for easing the process of creating a machine learning model and using the ML model for prediction. This application aims to help non-experts users experience the journey of how a machine learning model is built without requiring them to code. The application has three main menus: EDA, Modelling, and Prediction.

## Link to running app

https://hanifnoerr-codelessml-about-5p90s9.streamlit.app/

# User Manual

In the following you will find how they are supposed to be used.
- The EDA menu is used to explore and performs automatic visualisation of any dataset without writing a single code. Just upload your data in a compatible file format (csv or Excel).
- The Modelling menu is used to create machine learning models; you need to select the type of task you want to perform: Classification or Regression. This menu also performs data processing and perform an evaluation of all 9 ML algorithms available in this application. For now, only default data processing, such as dropping columns that have equal or more than 40% of missing values, dropping columns that are 100% unique, inputting missing values with their mean if the data type is number or mode if the data type is a string is available. These steps are necessary because most machine learning algorithms can't deal with missing values and are automatically executed when you click submit button on the Modelling menu.
- The Prediction menu is used to predict new data for which you do not know the target using a trained model.

## How to use

1. Explore and understand your data using the EDA menu (e.g., descriptive statistics, data visualisation, correlation matrix, etc.)
2. Go to the Modelling menu. [Modelling - Classification] is for the classification task, and [Modelling – Regression] is for the regression task. Choose the menus that best suit your problem. Then select the target variable and determine the ratio of data that will be used for training and testing your model. After that, click submit, and the application will automatically perform data processing and training ML models.
3. Determine which model you want to save and Click “Select model to download”. You will get model.joblib and encoder.pkl as well as the original data that have been encoded with label encoder inside a zipped folder.
4. Use the model.joblib and encoder.pkl to predict your new data in the “Prediction Menu” and save the result by clicking the “Downlaod Output File” button.

## Available machine learning algorithms

* Classification Task:

| Name                                	|          Reference         	|
|--------------------------------------	|:--------------------------:	|
| Logistic Regression    	|        [[1]](#c1)        	|
| Linear SVC       	|        [[1]](#c1)       	|
| K-Nearest Neighbor       	|        [[1]](#c1)       	|
| Multinomial Naive Bayes                 	|  [[1]](#c1) 	|
| Decision Tree            	| [[1]](#c1) 	|
| Random Forest       	|  [[1]](#c1) 	|
| Gradient Boosting   	|        [[1]](#c1)        	|
| LightGBM      	|        [[1]](#c2)        	|
| XGBoost          |        [[1]](#c3)           |

## Installation on local machine

## Dataset

The following datasets were used for the development and testing of this application:
- Classification Task:
  - Aeberhard,Stefan & Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.
- Regression Task:
  - Schlimmer,Jeffrey. (1987). Automobile. UCI Machine Learning Repository. https://doi.org/10.24432/C5B01C.
  
# References

<a name="c1">**[1]**</a>Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
 
<a name="c1">**[2]**</aKe, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, 3146–3154.

<a name="c1">**[3]**</a>Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM.


# License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
