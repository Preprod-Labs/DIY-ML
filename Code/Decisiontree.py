# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harish S
        # Role: Architect
        #  Code ownership rights: Harish S
    # Version:
        # Version: V 1.0 (19 Mar 2024)
            # Developer: Harish S
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code enables data ingestion from an external data sources.
        # MYSQL: Yes
        # NoSQL: No
        # MQs: Implemented in a seprate code
        # Cloud: To be implemented
        # Data versioning: No
        # Data masking: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        #Python 3.10.13
        #Numpy 1.26.1
        #Pandas 2.2.1

#Import Libraries

import numpy as np #numerical computation library
import pandas as pd #data manipulation library
import sklearn  #scientific library for machine learning
from sklearn.tree import DecisionTreeRegressor #Decision tree class from sklearn
from sklearn.model_selection import train_test_split # to split the data into Train and test data 
from sklearn.metrics import r2_score,mean_squared_error # Metrics to evaluate the linear regression model
import pickle,joblib #libraries to save models

Path=input("Enter the path of dataset: ") # Enter the path of dataset
modelpath=input("Enter the path to save models: ") #Enter the path to save models

df=pd.read_csv(Path+"Mobileprice.csv") # Import and read the csv file

df_tree=df.copy(deep=True) # Copy the contents to another dataframe df_tree

#Preprocess the data for DECISION TREE MODEL

#Map the categorical features to  numerical

map_brand = {'Apple': 0, 'Oppo': 1, 'Xiaomi': 2, 'Samsung':3,'Oneplus':4,'Motorola':5}
df_tree['Brand_name']=df_tree['Brand_name'].map(map_brand).astype('category')

map_paytype = {'cash':0, 'credit card':1, 'debit card':2, 'paypal':3}
df_tree['Payment_type']=df_tree['Payment_type'].map(map_paytype).astype('category')

#Convert date values into granual features "day, month and year " and delete Purchase date feature
df_tree['Purchase_date']=pd.to_datetime(df_tree['Purchase_date'],errors='coerce')
df_tree['day']=df_tree['Purchase_date'].dt.day
df_tree['month']=df_tree['Purchase_date'].dt.month
df_tree['year']=df_tree['Purchase_date'].dt.year

#Drop features not required for model 
df_tree.drop(['Customer_id','Purchase_date'],axis=1,inplace=True)

#Select input and target features
y=df_tree['Price']
X=df_tree.drop('Price',axis=1)

#Split the data into train and test
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)

# DEFINE DECISION TREE REGRESSION CLASS
dt_model=DecisionTreeRegressor(max_depth=3,max_leaf_nodes=2)

#FIT THE MODEL ON TRAIN DATA
dt_model.fit(Xtrain,ytrain)

#PREDICT MODEL OUTPUTS ON TEST DATA
dt_yprd=dt_model.predict(Xtest)

#EVALUATE THE MODEL OUTPUTS YPRED AND YACTUAL (YTEST) USING METRICS
dt_mse = mean_squared_error(ytest,dt_yprd)
dt_r2 = r2_score(ytest,dt_yprd)

#PRINT THE METRICS
print("mse of lr model:",dt_mse)
print("r2 of lr model:",dt_r2)

#Save the model

with open(modelpath+'dtmodel.sav', 'wb') as f:
    pickle.dump(dt_model, f)    