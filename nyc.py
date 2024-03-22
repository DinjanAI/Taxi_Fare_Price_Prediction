import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

df_train = pd.read_csv('E:/python/predict texi/predict/taxi_fare/train.csv', nrows=50_000)
df_test = pd.read_csv('E:/python/predict texi/predict/taxi_fare/test.csv')
df_train.info()

# null value
nan_train = pd.DataFrame(data=df_train.isnull().sum(), columns=['Train NaN'])
nan_test = pd.DataFrame(data=df_test.isnull().sum(), columns=['Test NaN'])
nan_test.loc['total_fare'] = 0
result = pd.concat([nan_train, nan_test], axis=1, sort=False)
print(result)

df_train.dropna(inplace=True)
# statistical information
description = df_train.describe()
print(description)

def clear_fare(df):
    # Fare interval
    min_fare, max_fare = 1.50, 100
    # Applying the limits
    true_fare = df['total_fare'].between(min_fare, max_fare)
    return df[true_fare]
df_train = clear_fare(df_train)
print(df_train.head())

plt.figure(figsize=(20,2))
sns.heatmap(df_train.corr(), cmap='rainbow', linecolor='black', linewidths=0.5,annot=True)
plt.show()

plt.figure(figsize=(12,6))
sns.distplot(df_train['total_fare'], bins=80, kde=False)
sns.despine(top=True, bottom=True, left=True, right=True)
plt.show()

for col in df_train.columns:
    if col != 'total_fare':
        sns.scatterplot(x=df_train['total_fare'], y=df_train[col])
        plt.show()
# Assuming grid_search is already defined and fitted
# Assuming features_on contains the list of features you want to use for prediction

# Splitting data into training and testing sets
X = df_train.drop(columns=['total_fare'])
y = df_train['total_fare']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # random_state for reproducibility

# Preprocessing for numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Bundle preprocessing for numerical and non-numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
y_pred = clf.predict(X_test)

# Creating a DataFrame to compare real and predicted fare values
comparison_df = pd.DataFrame({'Real Fare': y_test, 'Predicted Fare': y_pred})

# Calculating and displaying summary statistics
summary_stats = comparison_df.describe().T.drop('count', axis=1)
print(summary_stats)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')