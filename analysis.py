import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data_path ="/Users/hsiyenhsieh/Documents/GitHub/github_code/social-media-impact-on-suicide-rates.csv"
data = pd.read_csv(data_path)
"""
#exploring data
print(data.head())
print(data.info())
"""
#standardize columns names
data.columns = (data.columns.str.lower().str.replace(' ', '_').str.strip())
print(data.info())

#base on sex data analysis 
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='year', y='suicide_rate_%_change_since_2010', hue='sex', marker='o')
plt.title("Yearly Suicide Rate % Change Since 2010 by Sex")
plt.xlabel("Year")
plt.ylabel("Suicide Rate % Change")
plt.show()

#social media change 
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='year', y='twitter_user_count_%_change_since_2010', label="Twitter", marker='o')
sns.lineplot(data=data, x='year', y='facebook_user_count_%_change_since_2010', label="Facebook", marker='o')
plt.title("Social Media User Count % Change Since 2010")
plt.xlabel("Year")
plt.ylabel("User Count % Change")
plt.legend()
plt.show()

#regression analysis
X = data[['twitter_user_count_%_change_since_2010', 'facebook_user_count_%_change_since_2010']]
y = data['suicide_rate_%_change_since_2010']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fitting the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predictions and evaluation
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2 Score:", r2)
print("Mean Squared Error:", mse)
