import pandas as pd
from sklearn.model_selection import train_test_split
import common
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



RANDOM_STATE = 24


def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    
    data = pd.read_csv(path)
    X = data.drop(columns=['trip_duration'])
    y = data['trip_duration']
    return X, y


def fit_model(X, y):

    print(f"Fitting a model")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    num_features = [ 'hour']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
       ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
       ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
       
       ('ohe_and_scaling', column_transformer),
       ('regression', Ridge())
    ])

    model = pipeline.fit(X_train[train_features], y_train)
    y_pred_train = model.predict(X_train[train_features])
    y_pred_test = model.predict(X_test[train_features])

    print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
    print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
    print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))

    return model



if __name__ == "__main__":

    X_train, y_train = load_train_data(common.DB_PATH)
    
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)
    