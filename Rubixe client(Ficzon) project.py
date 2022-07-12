#!/usr/bin/env python
# coding: utf-8

# ## CLIENT PROJECT
# ### Client: Sales Effectiveness | Category: Product Sales
# ### Project Ref: PM-PR-0019

# ### Business case:
# FicZon Inc is an IT solution provider with products ranging from on-
# premises products to SAAS based solutions. FicZon major leads generation channel is digital and through their website.
# FicZon business is majorly dependent on the sales force
# effectiveness. As the market is maturing and more new competitors
# entering the market, FicZon is experiencing the dip in sales.
# Effective sales is dependent on lead quality and as of now, this is
# based on manual categorization and highly depended on sales staff.
# Though there is a quality process, which continuously updates the
# lead categorization, it’s value is in for post analysis, rather than
# conversation.
# FicZon wants to explore Machine Learning to pre-categorize the lead
# quality and as result, expecting significant increase in sales
# effectiveness.

# ### Project objectives: 
# 1. Data exploration insights – Sales effectiveness.
# 2. ML model to predict the Lead Category (High Potential , Low
# Potential)

# ### Database information:
# The dataset required for the given project was accessed using the following details:
# DB Name: project_sales
# Table Name: data
# Host: 18.136.157.135
# Port: 3306
# Username: dm_team2
# Password: DM!$Team&27@9!20!
# The acquire dataset had 7422 entries and 9 features:
# - Created 
# - Product_ID
# - Source 
# - Mobile 
# - EMAIL 
# - Sales_Agent 
# - Location 
# - Delivery_Mode 
# - Status

# ## STEP 1: 
# ### Importing necessary libraries and packages

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')


# ## STEP 2: 
# ### Loading the dataset

# In[2]:


database_name ='project_sales'
Host='18.136.157.135'
Username='dm_team2'
Password='DM!$Team&27@9!20!'
connect=create_engine('mysql+pymysql://'+Username+':'+Password+'@'+Host+'/'+database_name)


# In[3]:


connect.table_names()


# In[4]:


connect


# In[5]:


query='select*from data'
data=pd.read_sql(query,connect)


# ## STEP 3: 
# ### Exploratory data analysis

# In[6]:


data.head(10)


# ### Metadata

# In[7]:


data.info()


# In[8]:


for i in data.columns:
    print(data[i].unique())


# ### Checking null values

# In[9]:


data.replace('', np.nan,inplace=True)


# In[10]:


data.isnull().sum()


# ## STEP 4:
# ## Data pre-processing
# Data pre-processing is required for further exploratory data analysis and for training the machine learning models to be tested.

# #### Dropping null values

# In[11]:


data=data.dropna()


# In[12]:


data.head(20)


# In[13]:


data.isnull().sum()


# #### Dropping the irrelevant variables

# As 'EMAIL','Mobile' and 'Created' features will not affect the lead quality in anyway they can be eliminated from the dataset

# In[14]:


data.drop(['EMAIL','Mobile','Created'],axis=1,inplace=True)


# In[15]:


data.head(10)


# In[17]:


data.shape


# In[18]:


data.info()


# In[19]:


data.reset_index(drop=True,inplace=True)


# #### Encoding the categorical variables
# All the features given are categorical variables and need to be encoded.

# In[20]:


encoder=OrdinalEncoder()


# In[21]:


data=pd.DataFrame(encoder.fit_transform(data),columns=data.columns)


# In[22]:


encoder.categories_


# In[23]:


data


# ##### There is no feature which indicate the category to which a lead belongs, instead the staus of these leads are given which can be used to logically interpret whether the lead is a 'high potential' or a 'low potential' one.

# In[24]:


status=list(zip(list(range(11)),encoder.categories_[5]))


# In[25]:


status=pd.DataFrame(status,columns=['encode values','status'])


# In[26]:


status


# ##### The features 'Open','Potential','In Progress Positive','CONVERTED', and 'Long Term' can be categorised as high potential leads and  'Not Responding','Just Enquiry','Junk Lead','In Progress Negative', and 'LOST' can be classified under low potential leads.

# In[27]:


positive_status=[0,2,6,8,9,10]
negative_status=[1,3,4,5,7]


# In[28]:


data.head()


# In[29]:


high_potential=pd.DataFrame()
for i in positive_status:
    a=data[data['Status']==i]
    high_potential=pd.concat([high_potential,a])


# In[30]:


high_potential


# In[31]:


high_potential.reset_index(inplace=True,drop=True)


# In[32]:


low_potential=pd.DataFrame()
for i in negative_status:
    a=data[data['Status']==i]
    low_potential=pd.concat([low_potential,a])


# In[33]:


low_potential


# In[34]:


low_potential.reset_index(inplace=True,drop=True)


# ##### The high-potential category is assigned the value 0 and the low potential category is given the value 1

# In[35]:


h=[]
for i in range(2237):
    h.append(0)


# In[36]:


h=pd.DataFrame(h,columns=['category'])


# In[37]:


high_potential=high_potential.join(h,how='right')


# In[38]:


l=[]
for i in range(3294):
    l.append(1)


# In[39]:


l=pd.DataFrame(l,columns=['category'])


# In[40]:


low_potential=low_potential.join(l,how='right',rsuffix='')


# In[41]:


data_new=pd.concat([high_potential,low_potential])


# In[42]:


data_new


# #### Dropping the Status feature
# This is done as the data points has already been categorised into 'high potential'(0) and 'low potential'(1) ones using the status values.

# In[43]:


data_new.drop('Status',axis=1,inplace=True)


# #### Shuffling the data points

# In[44]:


data_new=data_new.sample(frac=1)


# In[45]:


data_new.reset_index(inplace=True,drop=True)


# In[46]:


data_new.head()


# ### Determining the Feature importances
# This is done using the SelectKBest package from scikitlearn using chi2 as the scoring function,

# In[47]:


from sklearn.feature_selection import SelectKBest,chi2


# In[48]:


x=data_new.iloc[:,:5]
y=data_new.iloc[:,5]


# In[49]:


x


# In[50]:


y


# In[51]:


fetr=SelectKBest(score_func=chi2,k='all')
fit=fetr.fit(x,y)


# In[52]:


fit.scores_


# In[53]:


import matplotlib.pyplot as plt


# In[54]:


plt.figure(figsize=(18,10))
plt.barh(x.columns, fit.scores_,color='r')
for i,v in enumerate(fit.scores_):
    plt.text(v,i,str(v),fontweight=1000,fontsize=15)


# As obvious Location is the factor which influences the lead category the most. This is followed by Delivery mode and Product occupying the second and third position respectively.

# ### Exploring feature wise distribution of data

# In[55]:


for i in x.columns:
    print('number of unique values in',i,':',len(data_new[i].unique()))


# In[56]:


encoder.categories_


# In[57]:


plt.figure(figsize=(20,19))
for i,v in list(zip(list(range(1,30)),data_new['Product_ID'].unique())):
    plt.subplot(5,6,i)
    sns.countplot(data_new[data_new['Product_ID']==v]['category'],palette='deep')
    plt.xlabel('product'+encoder.categories_[0][int(v)],fontweight=1000,fontsize=12)


# #### Insights from the countplot for product_ID
# - The leads associated with the products 1,8,15,22 are very less likely to be high-potential leads
# - The leads for the products 2,5,12,11,25,and 27 have comparitively higher chances of being low potential
# - Leads for product 14 are very much likely to be high potential leads
# - Leads for product 21 are very much likely to be high potential leads
# - The leads for the products 3,9,10,18,19,and 20 have comparitively higher chances of being high potential
# - Products 0,13,16,17,23,and 24 have too less number of  sales to decide upon the probable category to which their leads belong
#  

# In[58]:


plt.figure(figsize=(20,20))
for i,v in list(zip(list(range(1,23)),data_new['Source'].unique())):
    plt.subplot(5,5,i)
    sns.countplot(data_new[data_new['Source']==v]['category'],palette='colorblind')
    plt.xlabel(encoder.categories_[1][int(v)],fontweight=1000,fontsize=12)


# ### Insights from countplots for source
# 1. Contacts made through the following channels have comparitively higher probability of being low potential
# - Live chat-direct
# - call
# - website
# - live chat-PPC
# - just dial
# - live chat-blog
# - live chat-CPC
# - live chat-Google ads
# 2. Leads made through customer referral and by recommendation are very much likely to be high potential ones
# 3. Obviously existing clients and customers are high potential leads of the company
# 4. Leads through US website are more likely to be low potential
# 5. Only few leads are generated through campaign,e-mail message and CRM form

# In[59]:


plt.figure(figsize=(20,15))
for i,v in list(zip(list(range(1,12)),data_new['Sales_Agent'].unique())):
    plt.subplot(4,3,i)
    sns.countplot(data_new[data_new['Sales_Agent']==v]['category'],palette='dark')
    plt.xlabel(encoder.categories_[2][int(v)],fontweight=1000,fontsize=15)


# ### Insights from countplots for sales agents
# - The following agents have generated more number of high potential leads and hence are more efficient:
#     - sales agent 6
#     - sales agent 8
#     - sales agent 9
# - The following agents have generated more number of low potential leads and hence are less efficient:
#     - sales agent 3
#     - sales agent 4
#     - sales agent 5
#     - sales agent 11
#     - sales agent 12
# - sales agents 2 and 7 have generated more or less equal number of low and high potential leads
# - sales agent 10 have only made a few leads, hence the efficiency cannot be determined as of now

# In[60]:


plt.figure(figsize=(18,15))
for i,v in list(zip(list(range(1,17)),data_new['Location'].unique())):
    plt.subplot(4,4,i)
    sns.countplot(data_new[data_new['Location']==v]['category'],palette='rainbow')
    plt.xlabel(encoder.categories_[3][int(v)],fontweight=1000,fontsize=15)


# ### Insight from countplots for lead location
# - leads from Kolkata, Delhi, Mumbai, Pune, Trivandrum, Hyderabad, US and Australia have higher probability of being low           potential
# - leads from bangalore are more likely to be high potential
# - The lead categorisation for locations like UAE, UK, Singapore and Chennai cannot be certain as the numbers of high and low     potential leads are nearly the same
# - Very few leads are made from Europe and Malaysia
# - Leads from other locations are very likely to be low potential

# In[61]:


plt.figure(figsize=(20,15))
for i,v in list(zip(list(range(1,6)),data_new['Delivery_Mode'].unique())):
    plt.subplot(4,5,i)
    sns.countplot(data_new[data_new['Delivery_Mode']==v]['category'],palette='bright')
    plt.xlabel(encoder.categories_[4][int(v)],fontweight=1000,fontsize=15)


# ### Insights from countplots for delivery modes
# - Leads associated with the delivery modes 3 and 5 are likely to be low potential
# - Leads associated with the delivery modes 1 and 4 are likely to be high potential
# - very few leads are associated with delivery mode 2

# ### Checking multicollinearity

# In[62]:


sns.pairplot(data_new,hue='category')


# No strong correlations are observed between the features of the dataset, hence feature eliminations are not required.

# ### Checking imbalance in data

# In[63]:


import seaborn as sns
sns.countplot(data_new['category'])


# The target variable values are imbalanced in the dataset.

# ### Splitting dataset into training and testing data

# In[70]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# ### Adressing the imbalance using SMOTE

# In[71]:


from imblearn.over_sampling import SMOTE
oversampler=SMOTE()
x_train,y_train=oversampler.fit_resample(x_train,y_train)


# In[72]:


sns.countplot(y_train)


# ## Step 5:
# ### Building different Machine learning models and testing their efficiencies

# In[73]:


logr=LogisticRegression()
knn=KNeighborsClassifier(n_neighbors=3)
forest=RandomForestClassifier(max_depth=4)
adab=AdaBoostClassifier()
xgb=XGBClassifier(max_depth=4)


# In[74]:


logr.fit(x_train,y_train)


# In[75]:


knn.fit(x_train,y_train)


# In[76]:



forest.fit(x_train,y_train)


# In[77]:


adab.fit(x_train,y_train)


# In[78]:


xgb.fit(x_train,y_train,verbose=False)


# In[79]:


model_list=[logr,knn,forest,adab,xgb]


# In[80]:


colors=['r','g','b','y','m']
labels=['logistic regression','K nearest neighbour ','Random forest','Adaboost','Xgboost']


# In[81]:


for i in model_list:
    print('accuracy score for',i,'model is')
    print(accuracy_score(y_true=y_test,y_pred=i.predict(x_test.values)))


# In[84]:


a=0
b=0
plt.figure(figsize=(10,8))
for i in model_list:
    fpr,tpr,thresholds=roc_curve(y_test,i.predict_proba(x_test.values)[:,1])
    plt.plot(fpr,tpr,color=colors[a],label=labels[b])
    a=a+1
    b=b+1
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
    


# In[85]:


for i in model_list:
    print('auc for',i,'model is')
    print(roc_auc_score(y_test,i.predict_proba(x_test)[:,1]))


# From the above observations, it is clear that the xgboost classifier is more efficient for this classification problem.

# ## Step 6:
# ### Hyper-parameter tuning

# From the above results it can be concluded tht among the various classification model the best one whic suits the problem in hand is the XGBoost classifier model. Inorder to further improve the accuracy hyperparameter tuning is done to find out the best combination of parameters for the XGBoost model.

# In[86]:


parameters={'max_depth':[1,2,3,4,5,6,7,8,9,10],
            'n_estimators':[100,150,200,250,300,350,400,450,500,550,600,650,700,750,
                            800,850,900,950,1000,1500,2000],
            'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
model=XGBClassifier()
grid=GridSearchCV(model,parameters,cv=4)


# In[87]:


grid.fit(x,y)


# In[88]:


grid.best_params_


# In[89]:


grid.best_score_


# In[101]:


best_model=grid.best_estimator_


# ## Step 7:
# ### Testing the best model

# In[104]:


predictions=best_model.predict(x_test)


# In[105]:


accuracy_score(y_true=y_test,y_pred=predictions)


# ## Step 8:
# ### Saving the best model

# In[106]:


joblib.dump(best_model,'Ficzon_lead_prediction_model')


# ## Project summary
# The main objectives of this project was to form data exploration insights from the sales data of the enterprise called FicZon to better understand the sales effectiveness and to build a machine learning model to pre-categorize the leads generated in the company as high or low potential ones.
# - Just like any other data science project the initial step was to import the necessary packages. 
# - The database details provided were used to load in the dataset using create_engine function from sqlalchemy.
# - The next step was to carry out the exploratory data analysis, where at first, all the necessary metadata was obtained.
#     - The dataset had null values and some irrelevant features
#     - All the features were categorical variables
#     - Target variable coloumn(high and low potential) was not provided, instead a status coloumn was given
# - Data pre-processing was required to further progress the exploratory data analysis and to prepare the data for training the     machine learning models.
#     - Data points with null values were dropped
#     - Irrelevant variables were dropped
#     - All the categorical variables were encoded using OrdinalEncoder from scikit learn
#     - Status values were used to classify the data points into 'high potential'(0) and 'low potential'(1) leads
#     - Status variable was dropped after the above operation
#     - Feature importances were checked using SelectKBest package from scikit learn using chi2 as the score function
#     - Feature wise distribution of the target variable (category) values were determined by plotting countplots using seaborn         and necessary insights were delineated from them
#     - Pairplot function from seaborn was used to check multicollinearity in the dataset and was found to be none
#     - Target variable values were found to be imbalanced and this was addressed using SMOTE package from imblearn api
# - Next step in the workfolw was to build the possible machine learning models and select the best one among them for the         problem in hand.
#     - Since there is no single perfect model for all the prediction tasks, all the classification algorithms including Logistic       regression, Random forest classifier, XGBoost classifier, AdaBoost classifier and K-nearest Neighbour classifer, were           tried out and the most effective one was chosen by plotting the receiver operator curves(ROC) and evaluating the Area           under the curve values(AUC) for each of the models.
# - XGBclassifier was found to be the most effective in prediction and hyperparameter tuning for the same was carried out using     the GridSearchCV function from scikit learn. The best model was tested and saved using the joblib package.
# 
