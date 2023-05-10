# CodelessML
![CodelessML](https://github.com/hanifnoerr/CodelessML/blob/main/CodelessML.png?raw=true)
A web-based application for anyone who wants to learn and create machine learning model without coding.

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
2. Go to the Modelling menu. [Modelling - Classification] for the classification task, and [Modelling – Regression] for the regression task. Choose the menu that best suit your problem then select the target variable and determine the ratio of data that will be used for training and testing your model. click submit, and the application will automatically perform data processing and training ML models.
3. Determine which model you want to save and Click “Select model to download”. You will get model.joblib and encoder.pkl as well as the original data that have been encoded with label encoder inside a zipped folder.
4. Use the model.joblib and encoder.pkl to predict your new data in the “Prediction Menu” and save the result by clicking the “Downlaod Output File” button.

## Available machine learning algorithms

### Classification Task:

| Name                                	|          Reference         	|
|--------------------------------------	|:--------------------------:	|
| Logistic Regression    	|        [[1]](#c1)        	|
| Linear SVC       	|        [[1]](#c1)       	|
| K-Nearest Neighbor       	|        [[1]](#c1)       	|
| Multinomial Naive Bayes                 	|  [[1]](#c1) 	|
| Decision Tree            	| [[1]](#c1) 	|
| Random Forest       	|  [[1]](#c1) 	|
| Gradient Boosting   	|        [[1]](#c1)        	|
| LightGBM      	|        [[2]](#c2)        	|
| XGBoost          |        [[3]](#c3)           |

### Regression Task:

| Name                                	|          Reference         	|
|--------------------------------------	|:--------------------------:	|
| Linear Regression    	|        [[1]](#c1)        	|
| Support Vector Regression       	|        [[1]](#c1)       	|
| K-Nearest Neighbor       	|        [[1]](#c1)       	|
| Elastic Net                 	|  [[1]](#c1) 	|
| Passive Aggressive Regressor            	| [[1]](#c1) 	|
| Random Forest       	|  [[1]](#c1) 	|
| Gradient Boosting   	|        [[1]](#c1)        	|
| LightGBM      	|        [[2]](#c2)        	|
| XGBoost          |        [[3]](#c3)           |

## Installation 
This application is made using Streamlit, to deploy you can clone this repo and follow [this official guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

### Deploy on local machine
To deploy on local machine, you can use [Anaconda](https://www.anaconda.com/) and import [this environment](https://#).
Once your environment is ready, download this repo and run this following command
```
cd \to your CodelessML directory\
streamlit run About.py
```

## Dataset

The following datasets were used for the development and testing of this application:
- Classification Task: Wine Dataset [[4]](#c4)
- Regression Task: Automobile Dataset [[5]](#c5)
  
# References

<a name="c1">**[1]**</a> Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
 
<a name="c2">**[2]**</a> Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, 3146–3154.

<a name="c3">**[3]**</a> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM.

<a name="c4">**[4]**</a> Aeberhard,Stefan & Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

<a name="c5">**[5]**</a> Schlimmer,Jeffrey. (1987). Automobile. UCI Machine Learning Repository. https://doi.org/10.24432/C5B01C.

# License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
