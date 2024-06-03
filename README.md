# PreProd Corp DIY simple ML scripts for ML pipeline

This is a Repository of DIY-ML git feature branch

**Problem definition:**

1. Assume you are a data scientist and the data source is available at datalake ( Mongo DB)
2. Now you have json file as data source at hand ( Provided in git link ) extracted from data lake
3. Convert the json file to csv data to be used for EDA and ML modelling either by using Mongodb compass by import and export method OR using python 
4. Python code to convert JSON file is provided (TRY yourself -> JSONtoCSV.PY in CODE directory ) if not refer to csv file which is also provided (Mobileprice.csv) 
5. Execute scripts in CODE file to get the feel of BASIC ML pipeline to predict the Mobile phone prices ( Linearreg.py,Decisiontree.py and Randomforest.py )
6. save the appropriate model as per performance 

**Data definition:**

The dataset contains information on mobile phone sales of a US based ecommerce online start up company from year 2023 and 2024

**Data description:**

Customer_id: Unique identifier of a customer

Brand_name: Mobile phone brand names

Screen_size: the diagonal length of mobile phone

Battery_capacity: The amount of battery power available to run the mobile phone

Ram_size: the temporary memory size in mobile phone to run applications

Storage_capacity: the permanent memory size in mobile phone to run applications

Purchase_date: The date of customer purchase

Price: The price of the mobile phone

Payment_type: Describes how customer pays

Units_sold: The no of units sold  
