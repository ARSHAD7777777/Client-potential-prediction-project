# Client-potential-prediction-project
#### The main objectives of this project was to form data exploration insights from the sales data of the enterprise called FicZon to better understand the sales effectiveness and to build a machine learning model to pre-categorize the leads generated in the company as high or low potential ones.

#### Just like any other data science project the initial step was to import the necessary packages.
#### The database details provided were used to load in the dataset using create_engine function from sqlalchemy.
#### The next step was to carry out the exploratory data analysis, where at first, all the necessary metadata was obtained.
#### The dataset had null values and some irrelevant features
#### All the features were categorical variables
#### Target variable coloumn(high and low potential) was not provided, instead a status coloumn was given
#### Data pre-processing was required to further progress the exploratory data analysis and to prepare the data for training the machine learning models.
#### Data points with null values were dropped
#### Irrelevant variables were dropped
#### All the categorical variables were encoded using OrdinalEncoder from scikit learn
#### Status values were used to classify the data points into 'high potential'(0) and 'low potential'(1) leads
#### Status variable was dropped after the above operation
#### Feature importances were checked using SelectKBest package from scikit learn using chi2 as the score function
#### Feature wise distribution of the target variable (category) values were determined by plotting countplots using seaborn and necessary insights were delineated from them
#### Pairplot function from seaborn was used to check multicollinearity in the dataset and was found to be none
#### Target variable values were found to be imbalanced and this was addressed using SMOTE package from imblearn api
#### Next step in the workfolw was to build the possible machine learning models and select the best one among them for the problem in hand.
#### Since there is no single perfect model for all the prediction tasks, all the classification algorithms including Logistic regression, Random forest classifier, XGBoost classifier, AdaBoost classifier and K-nearest Neighbour classifer, were tried out and the most effective one was chosen by plotting the receiver operator curves(ROC) and evaluating the Area under the curve values(AUC) for each of the models.
#### XGBclassifier was found to be the most effective in prediction and hyperparameter tuning for the same was carried out using the GridSearchCV function from scikit learn. The best model was tested and saved using the joblib package.
