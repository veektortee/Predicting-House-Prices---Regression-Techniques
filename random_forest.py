import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '/Users/veek/Predicting-House-Prices---Regression-Techniques/train.csv'
df = pd.read_csv(iowa_file_path)
df.drop('Id', axis=1, inplace=True)
y = df['SalePrice']

# Select string-type features and one-hot encode 
string_df = df.select_dtypes(include=['object'])
one_hot = pd.get_dummies(df, columns=string_df.columns)

# Convert one-hot encoded features to int
for feature in one_hot.columns:
    if one_hot[feature].dtype == bool:
        one_hot[feature] = one_hot[feature].astype(int) 

# "cleaned" data
X = one_hot.drop(columns=['SalePrice'])

# Split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)
rf_val_predictions = rf_model.predict(X_val)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_val)
print(rf_val_mae)