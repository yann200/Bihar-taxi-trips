import pandas as pd
RANDOM_STATE = 24

data = pd.read_csv("data/data_ny.csv")
data = data.drop(columns=['id'])
data = data.drop(columns=['dropoff_datetime'])
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])


 #Nouveau DataFrame vide
new_data = pd.DataFrame()
data['weekday'] = data['pickup_datetime'].dt.weekday
data['month'] = data['pickup_datetime'].dt.month
data['hour'] = data['pickup_datetime'].dt.hour



# Extraction des colonnes souhaitées
new_data[['weekday', 'month', 'hour', 'trip_duration']] = data[['weekday', 'month', 'hour', 'trip_duration']]


new_data.to_csv('data/data_process.csv', index=True)

print("Fichier CSV sauvegardé avec succès!")



""" 
from sklearn.model_selection import train_test_split

X = data.drop(columns=['trip_duration'])
y = data['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)


def step1_add_features(X):
  res = X.copy()
  res['weekday'] = res['pickup_datetime'].dt.weekday
  res['month'] = res['pickup_datetime'].dt.month
  res['hour'] = res['pickup_datetime'].dt.hour
  res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
  return res

X = step1_add_features(X)
X_train = step1_add_features(X_train)
X_test = step1_add_features(X_test """



























