#Import Libraries for the JSON to CSV conversion

import pandas as pd #data manipulation libraries
import json #library to handle json data types

path=input("Enter the path to Save the csv data : ")

# Load JSON data from a file
with open(path+'DIY-ML.MLdata.json', 'r') as f:
    data = json.load(f)

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Remove the " _ID " Column which is a default identifier of Mongo DB
df.drop("_id",axis=1,inplace=True)

# Save DataFrame to CSV
df.to_csv(path+'Mobileprice_fromjson.csv', index=False)


