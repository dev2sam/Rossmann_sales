from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Read data head
stores = pd.read_csv("store.csv")
stores.head()


# Visualize missing data patterns
msno.matrix(stores)
plt.show()

# use seaborn's heatmap
sns.heatmap(stores.isnull(), cbar=False)
plt.show()

# Box plot to visualize outliers
selected_columns_stores = stores[['CompetitionDistance', 'CompetitionOpenSinceMonth',
                                  'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']]
selected_columns_stores.boxplot(figsize=(10, 6), vert=False)
plt.title('Box Plot for Outliers Detection')
plt.xlabel('Values')
plt.show()

stores['CompetitionDistance'].describe()

# Check null values
null_values = stores.isnull().sum()
na_values = stores.isna().sum()
print(null_values)
print(na_values)

# Count number of contained values
stores.count()

# Encoding StoreType, Assortment and PromoInterval
stores['StoreType'].value_counts()
stores['Assortment'].value_counts()
stores['PromoInterval'].value_counts()

# Assigns a unique integer to each category
label_encoder = LabelEncoder()
stores['StoreType_encoded'] = label_encoder.fit_transform(stores['StoreType'])
stores['Assortment_encoded'] = label_encoder.fit_transform(
    stores['Assortment'])

# Since PromoInterval is not ordinal, we need to create a custom mapping
promo_interval_mapping = {"Jan,Apr,Jul,Oct": 1,
                          "Feb,May,Aug,Nov": 2, "Mar,Jun,Sept,Dec": 3}
stores['PromoInterval_encoded'] = stores['PromoInterval'].map(
    promo_interval_mapping)
stores['PromoInterval_encoded'].fillna(-1, inplace=True)
stores['PromoInterval_encoded'] = stores['PromoInterval_encoded'].astype(
    'int32')

stores.head()

# Replace corresponding rows with 0 where 'Promo2' is 0, for Promo2SinceWeek, Promo2SinceYear, PromoInterval
stores.loc[stores['Promo2'] == 0, [
    'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']] = 0

# Convert 'Promo2SinceWeek' and 'Promo2SinceYear' to int64
stores['Promo2SinceWeek'] = stores['Promo2SinceWeek'].astype('int64')
stores['Promo2SinceYear'] = stores['Promo2SinceYear'].astype('int64')

# Construct Promo2Date by combining week and year.
stores['Promo2Date'] = pd.to_datetime(stores['Promo2SinceYear'].astype(
    str) + stores['Promo2SinceWeek'].astype(str) + '1', format='%Y%U%w', errors='coerce')
stores.head()

# Check datatypes
stores.info()

# Plotting correlation matrix
stores_copy = stores[['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                      'Promo2SinceWeek', 'Promo2SinceYear', 'StoreType_encoded', 'Assortment_encoded', 'PromoInterval_encoded']]
correlation_matrix = stores_copy.corr()
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plotting data distribution
plt.figure(figsize=(12, 6))
for i, column in enumerate(stores_copy.columns[:-1], 1):
    plt.subplot(2, 4, i)
    sns.histplot(stores_copy[column], kde=True)
    plt.title(f'{column}')
plt.tight_layout()
plt.show()

# For CompetitionOpenSinceMonth and CompetitionOpenSinceYear, check assortment
missing_values_df = stores[stores['CompetitionOpenSinceMonth'].isnull(
) & stores['CompetitionOpenSinceYear'].isnull()]
missing_values_df['Assortment'].value_counts()

# For CompetitionOpenSinceMonth and CompetitionOpenSinceYear, check store type
missing_values_df = stores[stores['CompetitionOpenSinceMonth'].isnull(
) & stores['CompetitionOpenSinceYear'].isnull()]
missing_values_df['StoreType'].value_counts()

# Calculate the median values for each 'Assortment' category
grouped_median_assortment = stores.groupby(
    'Assortment')[['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].median()


def impute_median_by_assortment(row, median_values):
    """
    Function to impute missing values with the median based on Assortment.
    """
    if pd.isna(row['CompetitionOpenSinceMonth']):
        row['CompetitionOpenSinceMonth'] = median_values.loc[row['Assortment'],
                                                             'CompetitionOpenSinceMonth']
    if pd.isna(row['CompetitionOpenSinceYear']):
        row['CompetitionOpenSinceYear'] = median_values.loc[row['Assortment'],
                                                            'CompetitionOpenSinceYear']
    return row


stores = stores.apply(lambda row: impute_median_by_assortment(
    row, grouped_median_assortment), axis=1)

# filling empty CompetitionDistance, with median as their distribution is skewed and not normally distributed
stores['CompetitionDistance'].fillna(
    stores['CompetitionDistance'].median(), inplace=True)

# Convert month and year to integers
stores['CompetitionOpenSinceMonth'] = stores['CompetitionOpenSinceMonth'].astype(
    'int64')
stores['CompetitionOpenSinceYear'] = stores['CompetitionOpenSinceYear'].astype(
    'int64')
stores['CompetitionDistance'] = stores['CompetitionDistance'].astype('int64')

# Create a new feature 'CompetitionOpenSinceDate' by combining CompetitionOpenSinceMonth and CompetitionOpenSinceYear and assuming first day of month
stores['CompetitionOpenSinceDate'] = pd.to_datetime(stores['CompetitionOpenSinceYear'].astype(
    str) + stores['CompetitionOpenSinceMonth'].astype(str) + '1', format='%Y%U%w', errors='coerce')
stores.head()

# Reading train head
train = pd.read_csv("train.csv")
train.head()

# Encoding categorical variables
train['StateHoliday'] = train['StateHoliday'].astype(str)
train['StateHoliday_encoded'] = label_encoder.fit_transform(
    train['StateHoliday'])

# Check whether Date corresponds to the correct DayOfWeek
train['Date'] = pd.to_datetime(train['Date'])
train['Day_of_Week'] = train['Date'].dt.dayofweek + 1

train['DayOfWeek'] = train['DayOfWeek'].astype('int64')
train['Day_of_Week'] = train['Day_of_Week'].astype('int64')

train['AreEqual'] = train['DayOfWeek'] == train['Day_of_Week']
train[train['AreEqual'] == False]

# Box plot to visualize outliers
selected_columns_train = train[['DayOfWeek', 'Sales', 'Customers', 'Store']]
selected_columns_train.boxplot(figsize=(10, 6), vert=False)
plt.title('Box Plot for Outliers Detection')
plt.xlabel('Values')
plt.show()

# Check null values
null_values2 = train.isnull().sum()
na_values2 = train.isna().sum()
print(null_values2)
print(na_values2)

# Count number of contained value
train.count()

# Creating a new variable for whether it was a holiday or not
train['isHoliday'] = train['SchoolHoliday'].apply(lambda x: 1 if x == 0 else 0)

# Check datatypes
train.info()

# Plotting correlation matrix
train_copy = train[['DayOfWeek', 'Sales', 'Customers',
                    'Open', 'Promo', 'isHoliday', 'SchoolHoliday']]
correlation_matrix = train_copy.corr()
plt.figure(figsize=(15, 13))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plotting data distribution
plt.figure(figsize=(12, 6))
for i, column in enumerate(train_copy.columns[:-1], 1):
    plt.subplot(2, 3, i)
    sns.histplot(train_copy[column], kde=True)
    plt.title(f'{column}')
plt.tight_layout()
plt.show()

# merging train and stores
# stores = stores.drop(columns=['StoreType', 'Assortment'])
train = train.drop(columns=['Day_of_Week', 'AreEqual'])
merged_df = pd.merge(train, stores, on='Store')
merged_df.to_csv('merged_df.csv', encoding='utf-8', index=False)
merged_df.head()

merged_df.count()

# merging test and stores
test = pd.read_csv("test.csv")
merged_df_test = pd.merge(test, stores, on='Store')
merged_df_test.to_csv('merged_df_test.csv', encoding='utf-8', index=False)
merged_df_test.head()

merged_df_test.count()

merged_df.info()

# Plot daily sales over time, on date level
merged_df['Year'] = merged_df['Date'].dt.year

# Create line plot using Seaborn
plt.figure(figsize=(15, 6))
sns.lineplot(data=merged_df, x="Date", y="Sales",
             hue="Year", marker='o', linestyle='-')
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend(title='Year', loc='upper left')  # Add legend
plt.show()

# Plot monthly sales each year
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d/%m/%Y')

# Extract year and month
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month

# Set up the figure and axis
plt.figure(figsize=(12, 8))
ax = sns.lineplot(x='Month', y='Sales', hue='Year', data=merged_df, marker='o')

# Customize the plot
ax.set_title('Sales for Each Month for Each Year')
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
plt.legend(title='Year', loc='upper right')

# Show the plot
plt.show()

# Assuming your DataFrame is named merged_df
plt.figure(figsize=(12, 8))

# Group by CompetitionDistance and calculate the mean sales for each distance
average_sales_by_distance = merged_df.groupby('CompetitionDistance')[
    'Sales'].mean().reset_index()

# Line plot for average sales vs competition distance
sns.lineplot(x='CompetitionDistance', y='Sales',
             data=average_sales_by_distance, marker='o')

plt.title('Average Sales vs Competition Distance')
plt.xlabel('Competition Distance')
plt.ylabel('Average Sales')
plt.show()

# Average sales by store type
average_sales_by_storetype = merged_df.groupby(
    'StoreType')['Sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(average_sales_by_storetype['StoreType'],
        average_sales_by_storetype['Sales'], color='skyblue')
plt.title('Average Sales by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Sales')
plt.show()

# Average sales by assortment
average_sales_by_assortment = merged_df.groupby(
    'Assortment')['Sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(average_sales_by_assortment['Assortment'],
        average_sales_by_assortment['Sales'], color='skyblue')
plt.title('Average Sales by Assortment')
plt.xlabel('Assortment')
plt.ylabel('Average Sales')
plt.show()

# Sales for promo vs non promo days
avg_sales_promo = merged_df.groupby('Promo')['Sales'].mean()
plt.figure(figsize=(10, 5))
avg_sales_promo.plot(kind='bar')
plt.title('Average Sales for Promo vs Non-Promo Days')
plt.xlabel('Promo (0 = No Promo, 1 = Promo)')
plt.ylabel('Average Sales')
plt.xticks(ticks=[0, 1], labels=['No Promo', 'Promo'], rotation=0)
plt.show()

# Comparison of sales and customers for stores before competition was opened and sales after it opened
before_competition = merged_df[merged_df['Date']
                               < merged_df['CompetitionOpenSinceDate']]
after_competition = merged_df[merged_df['Date']
                              >= merged_df['CompetitionOpenSinceDate']]

total_sales_before = before_competition['Sales'].sum()
total_customers_before = before_competition['Customers'].sum()
total_sales_after = after_competition['Sales'].sum()
total_customers_after = after_competition['Customers'].sum()

plt.figure(figsize=(10, 6))
periods = ['Before Competition', 'After Competition']
sales_data = [total_sales_before, total_sales_after]
customers_data = [total_customers_before, total_customers_after]

plt.bar(periods, sales_data, label='Total Sales')
plt.bar(periods, customers_data, label='Total Customers', alpha=0.5)

plt.title('Comparison of Sales and Customers Before and After Competition Opened')
plt.xlabel('Period')
plt.ylabel('Total')
plt.legend()
plt.grid(axis='y')
plt.show()

#
competition_distance = merged_df['CompetitionDistance']
sales = merged_df['Sales']

plt.figure(figsize=(10, 6))
plt.scatter(competition_distance, sales, alpha=0.5)
plt.title('CompetitionDistance vs. Sales')
plt.xlabel('CompetitionDistance')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# average sales for non-holiday days compared to combined state and school holidays
non_holiday_sales = merged_df[merged_df['isHoliday'] == 0]['Sales'].mean()
holiday_sales = merged_df[merged_df['isHoliday'] == 1]['Sales'].mean()
state_holiday_sales = merged_df[merged_df['StateHoliday'] == 1]['Sales'].mean()
school_holiday_sales = merged_df[merged_df['SchoolHoliday'] == 1]['Sales'].mean(
)

labels = ['Non-Holiday', 'Combined Holiday', 'State Holiday', 'School Holiday']
average_sales = [non_holiday_sales, holiday_sales,
                 state_holiday_sales, school_holiday_sales]

plt.figure(figsize=(10, 6))
plt.bar(labels, average_sales)
plt.title('Average Sales Comparison for Different Holiday Types')
plt.xlabel('Holiday Type')
plt.ylabel('Average Sales')
plt.grid(axis='y')
plt.show()

# averages sales by PromoInterval
merged_df['PromoInterval'] = merged_df['PromoInterval'].astype(str)
average_sales_by_promo_interval = merged_df.groupby(
    'PromoInterval')['Sales'].mean().reset_index()
average_sales_by_promo_interval = average_sales_by_promo_interval.sort_values(
    by='Sales', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(average_sales_by_promo_interval['PromoInterval'],
        average_sales_by_promo_interval['Sales'])
plt.title('Average Sales by PromoInterval')
plt.xlabel('PromoInterval')
plt.ylabel('Average Sales')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Checking trends and seasonality
sns.catplot(data=merged_df, x="DayOfWeek", y="Sales", hue="Promo", kind="bar")
plt.show()

# Sales trend over the months and year
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Year'] = merged_df['Date'].dt.year

# Specify the order of months
month_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharey=True)

# Flatten the 3x2 array to a 1D array for easy indexing
axes = axes.flatten()

# Loop through each combination of Promo and Year
for i, (promo, year) in enumerate(merged_df[['Promo', 'Year']].drop_duplicates().values):
    ax = axes[i]
    sns.pointplot(data=merged_df[(merged_df['Promo'] == promo) & (
        merged_df['Year'] == year)], x="Month", y="Sales", hue="Promo2", order=month_order, ax=ax)
    ax.set_title(f"Year {year}, Promo {promo}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# XGBoost

df = pd.read_csv("merged_df.csv")

# Convert date fields to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
df['CompetitionOpenSinceDate'] = pd.to_datetime(df['CompetitionOpenSinceDate'])

# Calculate the duration of competition being open in months
df['CompetitionDurationMonths'] = (
    (df['Date'].dt.year - df['CompetitionOpenSinceDate'].dt.year) * 12 +
    (df['Date'].dt.month - df['CompetitionOpenSinceDate'].dt.month)
)

# Impute missing dates on Promo2Date with a placeholder value, e.g., '1900-01-01'
df['Promo2Date'].fillna('1900-01-01', inplace=True)

# Replace NaNs in 'CompetitionDurationMonths' with the median duration
df['CompetitionDurationMonths'].fillna(
    df['CompetitionDurationMonths'].median(), inplace=True)

# Drop the original date fields and other non-numeric or unnecessary columns
df = df.drop(['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval', 'CompetitionOpenSinceMonth',
             'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1)

# Convert datetime columns to numeric features
df['Date_Year'] = df['Date'].dt.year
df['Date_Month'] = df['Date'].dt.month
df['Date_Day'] = df['Date'].dt.day

df['CompetitionOpenSinceDate_Year'] = df['CompetitionOpenSinceDate'].dt.year
df['CompetitionOpenSinceDate_Month'] = df['CompetitionOpenSinceDate'].dt.month
df['CompetitionOpenSinceDate_Day'] = df['CompetitionOpenSinceDate'].dt.day

df['Promo2Date'] = pd.to_datetime(df['Promo2Date'])
df['Promo2Date_Year'] = df['Promo2Date'].dt.year
df['Promo2Date_Month'] = df['Promo2Date'].dt.month
df['Promo2Date_Day'] = df['Promo2Date'].dt.day

# Drop the original datetime columns
df = df.drop(['Date', 'CompetitionOpenSinceDate', 'Promo2Date'], axis=1)

X_sales = df.drop(['Sales', 'Customers'], axis=1)
y_sales = df['Sales']  # target for sales

# Split the data for Sales
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42)

# XGBoost model for Sales
xgb_model_sales = xgb.XGBRegressor(
    objective='reg:squarederror', random_state=42)
xgb_model_sales.fit(X_train_sales, y_train_sales)

# Predict on the test set for Sales
y_pred_sales = xgb_model_sales.predict(X_test_sales)

# Evaluate performance for Sales
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
rmse_sales = np.sqrt(mse_sales)
rmspe_sales = rmse_sales / np.mean(y_test_sales) * 100

print(f'Mean Squared Error for Sales: {mse_sales}')
print(f'Root Mean Squared Error for Sales: {rmse_sales}')
print(f'Root Mean Squared Percentage Error for Sales: {rmspe_sales}')

X_customers = df.drop(['Sales', 'Customers'], axis=1)
y_customers = df['Customers']  # target for customers

# Split the data for Customers
X_train_customers, X_test_customers, y_train_customers, y_test_customers = train_test_split(
    X_customers, y_customers, test_size=0.2, random_state=42)

# XGBoost model for Customers
xgb_model_customers = xgb.XGBRegressor(
    objective='reg:squarederror', random_state=42)
xgb_model_customers.fit(X_train_customers, y_train_customers)

# Predict on the test set for Customers
y_pred_customers = xgb_model_customers.predict(X_test_customers)

# Evaluate performance for Customers
mse_customers = mean_squared_error(y_test_customers, y_pred_customers)
rmse_customers = np.sqrt(mse_customers)
rmspe_customers = rmse_customers / np.mean(y_test_customers) * 100

print(f'Mean Squared Error for Customers: {mse_customers}')
print(f'Root Mean Squared Error for Customers: {rmse_customers}')
print(f'Root Mean Squared Percentage Error for Customers: {rmspe_customers}')

# Get feature importances
importances_sales = xgb_model_sales.feature_importances_

# Assuming df is your DataFrame with relevant columns
features = X_sales.columns  # Assuming X_sales is your feature matrix

# Create a dictionary to map feature names to importances for Sales
feature_importance_sales_dict = dict(zip(features, importances_sales))

# Sort feature importances for Sales
sorted_feature_importance_sales = sorted(
    feature_importance_sales_dict.items(), key=lambda x: x[1], reverse=True)

# Convert the sorted feature importances to a DataFrame
df_feature_importance_sales = pd.DataFrame(
    sorted_feature_importance_sales, columns=['Feature', 'Importance'])

# Display or save the DataFrame
print(df_feature_importance_sales)


# Random Forest Regressor

df = pd.read_csv("merged_df.csv")
df.describe()

# Convert date fields to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
df['CompetitionOpenSinceDate'] = pd.to_datetime(df['CompetitionOpenSinceDate'])

# Calculate the duration of competition being open in months
df['CompetitionDurationMonths'] = (
    (df['Date'].dt.year - df['CompetitionOpenSinceDate'].dt.year) * 12 +
    (df['Date'].dt.month - df['CompetitionOpenSinceDate'].dt.month)
)

# Impute missing dates on Promo2Date with a placeholder value, e.g., '1900-01-01'
df['Promo2Date'].fillna('1900-01-01', inplace=True)

# Replace NaNs in 'CompetitionDurationMonths' with the median duration
df['CompetitionDurationMonths'].fillna(
    df['CompetitionDurationMonths'].median(), inplace=True)

# Drop the original date fields and other non-numeric or unnecessary columns
df = df.drop(['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval', 'CompetitionOpenSinceMonth',
             'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1)

# --> impute median by store type within each year, tell why you used storetype. then once you have the months categorize them in ranges
# --> also remove the thing you did with competitionDate and promo2Date
# --> CompetitionDistanceCategory, show outliers graph

# Convert datetime columns to numeric features
df['Date_Year'] = df['Date'].dt.year
df['Date_Month'] = df['Date'].dt.month
df['Date_Day'] = df['Date'].dt.day

df['CompetitionOpenSinceDate_Year'] = df['CompetitionOpenSinceDate'].dt.year
df['CompetitionOpenSinceDate_Month'] = df['CompetitionOpenSinceDate'].dt.month
df['CompetitionOpenSinceDate_Day'] = df['CompetitionOpenSinceDate'].dt.day

df['Promo2Date'] = pd.to_datetime(df['Promo2Date'])
df['Promo2Date_Year'] = df['Promo2Date'].dt.year
df['Promo2Date_Month'] = df['Promo2Date'].dt.month
df['Promo2Date_Day'] = df['Promo2Date'].dt.day

# Drop the original datetime columns
df = df.drop(['Date', 'CompetitionOpenSinceDate', 'Promo2Date'], axis=1)

# Define the target variables and features
targets = ['Sales', 'Customers']
features = df.drop(targets, axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, df[targets], test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor models
rf_sales = RandomForestRegressor(n_estimators=100, random_state=42)
rf_customers = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the RandomForestRegressor model for sales
rf_sales.fit(X_train, y_train['Sales'])

# Train the RandomForestRegressor model for customers
rf_customers.fit(X_train, y_train['Customers'])

# Predict on test data
sales_predictions = rf_sales.predict(X_test)
customers_predictions = rf_customers.predict(X_test)

# Evaluate performance for Sales
mse_sales = mean_squared_error(y_test['Sales'], sales_predictions)
rmse_sales = np.sqrt(mse_sales)
rmspe_sales = rmse_sales / np.mean(y_test['Sales']) * 100

print(f'Mean Squared Error for Sales: {mse_sales}')
print(f'Root Mean Squared Error for Sales: {rmse_sales}')
print(f'Root Mean Squared Percentage Error for Sales: {rmspe_sales}')

# Evaluate performance for Customers
mse_customers = mean_squared_error(y_test['Customers'], customers_predictions)
rmse_customers = np.sqrt(mse_customers)
rmspe_customers = rmse_customers / np.mean(y_test['Customers']) * 100

print(f'Mean Squared Error for Customers: {mse_customers}')
print(f'Root Mean Squared Error for Customers: {rmse_customers}')
print(f'Root Mean Squared Percentage Error for Customers: {rmspe_customers}')

# Get feature importances
feature_importances = rf_sales.feature_importances_
feature_importances2 = rf_customers.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame(
    {'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df2 = pd.DataFrame(
    {'Feature': X_train.columns, 'Importance': feature_importances2})
importance_df2 = importance_df2.sort_values(by='Importance', ascending=False)

# Print the top features
print("Top Features:")
print(importance_df.head())
print(importance_df2.head())

# Ridge regression

df = pd.read_csv("merged_df.csv")
df.describe()

# Convert date fields to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
df['CompetitionOpenSinceDate'] = pd.to_datetime(df['CompetitionOpenSinceDate'])

# Calculate the duration of competition being open in months
df['CompetitionDurationMonths'] = (
    (df['Date'].dt.year - df['CompetitionOpenSinceDate'].dt.year) * 12 +
    (df['Date'].dt.month - df['CompetitionOpenSinceDate'].dt.month)
)

# Replace NaNs in 'CompetitionDurationMonths' with the median duration
df['CompetitionDurationMonths'].fillna(
    df['CompetitionDurationMonths'].median(), inplace=True)

# Drop the original date fields and other non-numeric or unnecessary columns
df = df.drop(['Date', 'CompetitionOpenSinceDate', 'Promo2Date'], axis=1)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=[
                    'DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])

# drop sales and customers as they should not be part of the model
X = df.drop(['Sales', 'Customers'], axis=1)

# Define a range of alpha values to search
alphas = [0.1, 1, 10, 100]  # Add more values based on your requirements

# Define target for Sales
y_sales = df['Sales']

# Split the data for Sales
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X, y_sales, test_size=0.2, random_state=42)

# Standardize the features for Sales
scaler_sales = StandardScaler()
X_train_sales_scaled = scaler_sales.fit_transform(X_train_sales)
X_test_sales_scaled = scaler_sales.transform(X_test_sales)

# Create and fit RidgeCV model for Sales using standardized features
ridge_cv_model_sales = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv_model_sales.fit(X_train_sales_scaled, y_train_sales)

# Access the optimal alpha for Sales
optimal_alpha_sales = ridge_cv_model_sales.alpha_
print(f'Optimal Alpha for Sales: {optimal_alpha_sales}')

# Predict on the test set for Sales
y_pred_sales = ridge_cv_model_sales.predict(X_test_sales_scaled)

# Evaluate performance for Sales
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
rmse_sales = np.sqrt(mse_sales)
rmspe_sales = rmse_sales / np.mean(y_test_sales) * 100

print(f'Mean Squared Error for Sales: {mse_sales}')
print(f'Root Mean Squared Error for Sales: {rmse_sales}')
print(f'Root Mean Squared Percentage Error for Sales: {rmspe_sales}')

# Define features and target for Customers
y_customers = df['Customers']

# Split the data for Customers
X_train_customers, X_test_customers, y_train_customers, y_test_customers = train_test_split(
    X, y_customers, test_size=0.2, random_state=42)

# Standardize the features for Customers
scaler_customers = StandardScaler()
X_train_customers_scaled = scaler_customers.fit_transform(X_train_customers)
X_test_customers_scaled = scaler_customers.transform(X_test_customers)

# Create and fit RidgeCV model for Customers using standardized features
ridge_cv_model_customers = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv_model_customers.fit(X_train_customers_scaled, y_train_customers)

# Access the optimal alpha for Customers
optimal_alpha_customers = ridge_cv_model_customers.alpha_
print(f'Optimal Alpha for Customers: {optimal_alpha_customers}')

# Predict on the test set for Customers
y_pred_customers = ridge_cv_model_customers.predict(X_test_customers_scaled)

# Evaluate performance for Customers
mse_customers = mean_squared_error(y_test_customers, y_pred_customers)
rmse_customers = np.sqrt(mse_customers)
rmspe_customers = rmse_customers / np.mean(y_test_customers) * 100

print(f'Mean Squared Error for Customers: {mse_customers}')
print(f'Root Mean Squared Error for Customers: {rmse_customers}')
print(f'Root Mean Squared Percentage Error for Customers: {rmspe_customers}')
