import pandas as pd
import numpy as np
from celery import Celery
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import pickle
import os


model2task = {'LogisticRegression': 'classification',
              'SVC': 'classification',
              'DecisionTreeClassifier': 'classification',
              'RandomForestClassifier': 'classification',
              'Ridge': 'regression',
              'SVR': 'regression',
              'DecisionTreeRegressor': 'regression',
              'RandomForestRegressor': 'regression'}


def get_model(model, params):
    if model == 'LogisticRegression':
        return LogisticRegression(**params)
    elif model == 'SVC':
        return SVC(**params)
    elif model == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**params)
    elif model == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    elif model == 'Ridge':
        return Ridge(**params)
    elif model == 'SVR':
        return SVR(**params)
    elif model == 'DecisionTreeRegressor':
        return DecisionTreeRegressor(**params)
    elif model == 'RandomForestRegressor':
        return RandomForestRegressor(**params)


def calculate_metrics(estimator, X_train, X_test, y_train, y_test):
    """
    Calculate model performance metrics on both train and test set:
    - Accuracy, precision, recall, f1-score, and AUC-ROC for a classification task.
    - RMSE and MAE for a regression task.

    :param estimator: Sklearn estimator
    :param pd.DataFrame X_train: Train set
    :param pd.DataFrame X_test: Test set
    :param pd.DataFrame y_train: Train target
    :param pd.DataFrame y_test: Test target
    :return: Model metrics
    :rtype: dict
    """
    model = str(estimator).split('(')[0]
    metrics = {}
    if model2task[model] == 'classification':
        for x in ['train', 'test']:
            y_true = eval('y_' + x)
            y_pred = estimator.predict(eval('X_' + x))
            y_pred_proba = estimator.predict_proba(eval('X_' + x))
            metrics[x] = {}
            metrics[x]['accuracy'] = np.round(accuracy_score(y_true, y_pred), 4)
            average = 'binary' if len(set(y_true)) == 2 else 'macro'
            metrics[x]['precision'] = np.round(precision_score(y_true, y_pred, average=average), 4)
            metrics[x]['recall'] = np.round(recall_score(y_true, y_pred, average=average), 4)
            metrics[x]['f1_score'] = np.round(f1_score(y_true, y_pred, average=average), 4)
            metrics[x]['auc_roc'] = np.round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'), 4)
    else:
        for x in ['train', 'test']:
            y_true = eval('y_' + x)
            y_pred = estimator.predict(eval('X_' + x))
            metrics[x] = {}
            metrics[x]['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
            metrics[x]['mae'] = mean_absolute_error(y_true, y_pred)
    return metrics


CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)


@celery.task(name='train')
def train(model_id, model, data, params, grid_search, param_grid):
    """
    Train the given model with given parameters on the given data (json) and
    calculate its performance metrics.
    Save data, model and metrics into pkl files named by the model id.

    :param int model_id: Model id
    :param str model: Model to train
    :param json data: Data to fit and test the model
    :param str or dict params: Model parameters
    :param bool grid_search: Whether to perform grid search
    :param str or dict param_grid: Parameters grid for grid search
    :return: Task result
    :rtype: str
    """
    df = pd.read_json(data)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    estimator = get_model(model, params)

    if grid_search:
        gs = GridSearchCV(estimator, param_grid, cv=3, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)
        estimator = gs.best_estimator_

    estimator.fit(X_train, y_train)
    metrics = calculate_metrics(estimator, X_train, X_test, y_train, y_test)

    if model_id == 1:
        for dir in ['./data', './models', './metrics']:
            list(map(os.unlink, (os.path.join(dir, f) for f in os.listdir(dir))))

    with open(f'models/{model_id}.pkl', 'wb') as f:
        pickle.dump(estimator, f)
    with open(f'data/{model_id}_train.pkl', 'wb') as f:
        pickle.dump(df.loc[X_train.index], f)
    with open(f'data/{model_id}_test.pkl', 'wb') as f:
        pickle.dump(df.loc[X_test.index], f)
    with open(f'metrics/{model_id}.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return f'{model} is fitted and saved'


@celery.task(name='predict')
def predict(model_id):
    """
    Make model predictions on a train set and test set.

    :param int model_id: Model id to get predictions for
    :return: Predictions
    :rtype: dict
    """
    with open(f'models/{model_id}.pkl', 'rb') as f:
        estimator = pickle.load(f)
    with open(f'data/{model_id}_train.pkl', 'rb') as f:
        X_train = pickle.load(f).iloc[:, :-1]
    with open(f'data/{model_id}_test.pkl', 'rb') as f:
        X_test = pickle.load(f).iloc[:, :-1]

    train_pred = list(np.round(estimator.predict(X_train), 2))
    test_pred = list(np.round(estimator.predict(X_test), 2))

    return {'train_predictions': str(train_pred), 'test_predictions': str(test_pred)}


@celery.task(name='retrain')
def retrain(model_id, data):
    """
    Retrain the given model on a new training set and recalculate performance metrics.
    Save new train set, model and metrics into pkl files named by the model id.

    :param int model_id: Model id to retrain
    :param json data: Data to retrain
    :return: Task result
    :rtype: str
    """
    with open(f'models/{model_id}.pkl', 'rb') as f:
        estimator = pickle.load(f)
    with open(f'data/{model_id}_test.pkl', 'rb') as f:
        test = pickle.load(f)

    train_new = pd.read_json(data)
    X_train, y_train = train_new.iloc[:, :-1], train_new.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    estimator.fit(X_train, y_train)
    metrics = calculate_metrics(estimator, X_train, X_test, y_train, y_test)

    with open(f'data/{model_id}_train.pkl', 'wb') as f:
        pickle.dump(train_new, f)
    with open(f'models/{model_id}.pkl', 'wb') as f:
        pickle.dump(estimator, f)
    with open(f'metrics/{model_id}.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return f'Model {model_id} is re-fitted and saved'


@celery.task(name='delete')
def delete(model_id):
    """
    Delete a pkl file of the given model.

    :param int model_id: Model id to delete pkl for
    :return: Task result
    :rtype: str
    """
    os.remove(f'models/{model_id}.pkl')
    return f'Model {model_id} is deleted'


@celery.task(name='get_metrics')
def get_metrics(model_id):
    """
    Return performance metrics for the given model.

    :param int model_id: Model id to get performance metrics for
    :return: Train and test metrics
    :rtype: dict
    """
    with open(f'metrics/{model_id}.pkl', 'rb') as f:
        res = pickle.load(f)
    return res
