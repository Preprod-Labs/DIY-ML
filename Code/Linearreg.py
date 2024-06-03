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
from sklearn.preprocessing import StandardScaler # standard scaler library to bring all features on same scale
from sklearn.model_selection import train_test_split # to split the data into Train and test data 
from sklearn.linear_model import LinearRegression # Linear regression library 
from sklearn.metrics import r2_score,mean_squared_error # Metrics to evaluate the linear regression model

Path=input("Enter the path of dataset: ") # Enter the path of dataset

df=pd.read_csv(Path+"Mobileprice.csv") # Import and read the csv file

mod_df=df.copy(deep=True) # Copy the contents to another dataframe mod_df 

# EXTRACTING NUMERICAL FEATURES FOR LINEAR REGRESSION

df_linear=mod_df[['Screen_size', 'Battery_capacity',
       'Ram_size', 'Storage_capacity','Price','Units_sold']]

# DEFINE INPUT AND OUTPUT FEATURES

y=df_linear['Price']
X=df_linear.drop('Price',axis=1)

# SPLIT THE DATA 

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)

# SCALE THE INPUT FEATURES TO BRING VARYING FEATURES TO SAME SCALE

ss=StandardScaler()
scaler=ss.fit(Xtrain)
Xtrain_sd=ss.fit_transform(Xtrain)
Xtest_sd=ss.transform(Xtest)

# DEFINE LINEAR REGRESSION CLASS

lin_reg=LinearRegression()

#FIT THE MODEL ON TRAIN DATA
lin_reg.fit(Xtrain_sd, ytrain)

#PREDICT OUTPUTS ON TEST DATA
ypred_test  = lin_reg.predict(Xtest_sd)

#EVALUATE THE OUTPUTS YPRED AND YACTUAL (YTEST) USING METRICS

mse = mean_squared_error(ytest,ypred_test)
r2 = r2_score(ytest,ypred_test)

#PRINT THE METRICS
print("mse of lr model:",mse)
print("r2 of lr model:",r2)










