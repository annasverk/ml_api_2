import pandas as pd
import numpy as np
from celery import Celery
from flask import Flask, request
from flask_restx import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
import os
import warnings

warnings.filterwarnings('ignore')


model2task = {'LogisticRegression': 'classification',
              'SVC': 'classification',
              'DecisionTreeClassifier': 'classification',
              'RandomForestClassifier': 'classification',
              'Ridge': 'regression',
              'SVR': 'regression',
              'DecisionTreeRegressor': 'regression',
              'RandomForestRegressor': 'regression'}

model2grid = {'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100],
                                     'solver': ['newton-cg', 'lbfgs', 'sag'],
                                     'penalty': ['l2', 'none']},
              'SVC': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeClassifier': {'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'RandomForestClassifier': {'n_estimators': [100, 200, 300, 400, 500],
                                         'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'Ridge': {'alpha': np.linspace(0, 1, 11),
                        'fit_intercept': [True, False],
                        'solver': ['svd', 'lsqr', 'sag']},
              'SVR': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeRegressor': {'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]},
              'RandomForestRegressor': {'n_estimators': [100, 200, 300, 400, 500],
                                        'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]}}


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


CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']
celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

user = os.environ['POSTGRES_USER']
password = os.environ['POSTGRES_PASSWORD']
host = os.environ['POSTGRES_HOST']
port = os.environ['POSTGRES_PORT']
database = os.environ['POSTGRES_DB']
DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{user}'

app = Flask(__name__)
app.config['ERROR_404_HELP'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_CONNECTION_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
api = Api(app)
db = SQLAlchemy(app)


class MLModelsDAO:
    def __init__(self):
        self.ml_models = ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
                          'Ridge', 'SVR', 'DecisionTreeRegressor', 'RandomForestRegressor']
        self.ml_models_all = {}
        self.counter = 0  # total number of fitted models

    def get(self, model_id):
        """
        Return id of the celery prediction task for the given model.
        To get predictions, send a GET request to /results with a returned task id.
        Abort if the model was deleted.
        """
        if model_id not in self.ml_models_all:
            return f'Model {model_id} does not exist', 404
        if self.ml_models_all[model_id]['deleted']:
            return f'Model {model_id} was deleted', 404

        task = celery.send_task('predict', args=[model_id])
        return f'Task_id = {task.id}', 200

    def create(self, data, model, params, grid_search, param_grid):
        """
        Return id of the celery fitting task for the given data, model and parameters.
        If parameters are not set, use default values.
        If no exceptions, append model name and parameters to models_dao.ml_models_all dictionary.
        To check if the model is fitted, send a GET request to /results with a returned task id.
        """
        if model not in self.ml_models:
            return f'Can only train one of {self.ml_models} models', 404

        df = pd.read_json(data)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        if any(X.dtypes == object):
            return 'Could not support categorical features', 400
        if y.dtype == object and model2task[model] == 'regression':
            return f'{model} can only be used for regression tasks', 400
        elif y.dtype == float and model2task[model] == 'classification':
            return f'{model} can only be used for classification tasks', 400

        if params == 'default':
            params = {}
        if param_grid == 'default':
            param_grid = model2grid[model]
        if model == 'SVR' or model == 'SVC':
            params['probability'] = True

        try:
            estimator = get_model(model, params)
        except TypeError as err:
            return f'{model} got an unexpected keyword argument {str(err).split()[-1]}', 400

        if grid_search:
            if not all([param in estimator.get_params().keys() for param in param_grid]):
                return f'Invalid parameters for estimator {model}', 400

        self.counter += 1
        self.ml_models_all[self.counter] = {'model': model, 'params': estimator.get_params(),
                                            'retrained': False, 'deleted': False}
        task = celery.send_task('train', args=[self.counter, model, data, params, grid_search, param_grid])
        return f'Task_id = {task.id}', 200

    def update(self, model_id, data):
        """
        Return id of the celery prediction task for the given model on a new data.
        Abort if the model was deleted or a new data includes another columns.
        To get predictions, send a GET request to /results with a returned task id.
        """
        if model_id not in self.ml_models_all:
            return f'Model {model_id} does not exist', 404
        if self.ml_models_all[model_id]['deleted']:
            return f'Model {model_id} was deleted', 404

        self.ml_models_all[model_id]['retrained'] = True
        task = celery.send_task('retrain', args=[model_id, data])
        return f'Task_id = {task.id}', 200

    def delete(self, model_id):
        """
        Delete a pkl file of the given model.
        To check if the model is deleted, send a GET request to /results with a returned task id.
        """
        if model_id not in self.ml_models_all:
            return f'Model {model_id} does not exist', 404
        if self.ml_models_all[model_id]['deleted']:
            return f'Model {model_id} was already deleted', 404

        self.ml_models_all[model_id]['deleted'] = True
        celery.send_task('delete', args=[model_id])


models_dao = MLModelsDAO()


@api.route('/ml_api')
class MLModels(Resource):

    def get(self):
        """
        Return a list of models available for training.
        """
        return models_dao.ml_models

    def post(self):
        """
        Return id of the celery fitting task for the given data, model and parameters.
        If parameters are not set, use default values.
        If no exceptions, append model name and parameters to models_dao.ml_models_all dictionary.
        To check if the model is fitted, send a GET request to /results with a returned task id.
        """
        json_ = request.json
        data = json_['data']
        model = json_['model']
        params = json_.get('params', 'default')
        grid_search = json_.get('grid_search', False)
        param_grid = json_.get('param_grid', 'default')
        return models_dao.create(data, model, params, grid_search, param_grid)


@api.route('/ml_api/<int:model_id>')
class MLModelsID(Resource):

    def get(self, model_id):
        """
        Return id of the celery prediction task for the given model.
        To get predictions, send a GET request to /results with a returned task id.
        Abort if the model was deleted.
        """
        return models_dao.get(model_id)

    def put(self, model_id):
        """
        Return id of the celery re-fitting task for the given model on a new data.
        Abort if the model was deleted or a new data includes another columns.
        To get predictions, send a GET request to /results with a returned task id.
        """
        data = request.json
        return models_dao.update(model_id, data)

    def delete(self, model_id):
        """
        Delete a pkl file of the given model.
        To check if the model is deleted, send a GET request to /results with a returned task id.
        """
        models_dao.delete(model_id)
        return '', 204


@api.route('/ml_api/all_models')
class MLModelsAll(Resource):

    def get(self):
        """
        Return a dictionary of all fitted models and their parameters.
        """
        return models_dao.ml_models_all


@api.route('/results/<task_id>')
class Results(Resource):

    def get(self, task_id):
        """
        Return result for the given task id if task is completed and status otherwise.
        """
        res = celery.AsyncResult(task_id)
        if res.status == 'PENDING':
            return str(res.state), 200
        else:
            return res.get(), 200


class Metrics(db.Model):
    __tablename__ = 'metrics'
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer)
    data = db.Column(db.String(100))
    metric = db.Column(db.String(100))
    value = db.Column(db.Float)

    def __init__(self, model_id, data, metric, value):
        self.model_id = model_id
        self.data = data
        self.metric = metric
        self.value = value


@api.route('/metrics')
class DBMetrics(Resource):

    def get(self):
        """
        Database READ operation.
        Return a list of performance metrics for all the fitted models.
        """
        records = Metrics.query.all()
        output = []
        for record in records:
            output.append({'model_id': record.model_id, 'data': record.data,
                           'metric': record.metric, 'value': record.value})
        return output, 200


@api.route('/metrics/<int:model_id>')
class DBMetricsID(Resource):

    def post(self, model_id):
        """
        Database CREATE operation.
        Add performance metrics of the given model into a database.
        """
        task = celery.send_task('get_metrics', args=[model_id])
        result = celery.AsyncResult(task.id)
        res = result.get()
        for data in res:
            for metric in res[data]:
                value = np.round(res[data][metric], 4)
                model_metric = Metrics(model_id, data, metric, value)
                db.session.add(model_metric)
                db.session.commit()
        return f'Metrics for model {model_id} are added', 200

    def get(self, model_id):
        """
        Database READ operation.
        Return a list of performance metrics for the given model on both train and test set.
        """
        model_records = Metrics.query.filter_by(model_id=model_id).all()
        output = []
        for record in model_records:
            output.append({'data': str(record.data), 'metric': str(record.metric), 'value': str(record.value)})
        return output, 200

    def put(self, model_id):
        """
        Database UPDATE operation.
        Update performance metrics of the given model in a database.
        Should be used if the model was retrained on a new training set.
        """
        task = celery.send_task('get_metrics', args=[model_id])
        result = celery.AsyncResult(task.id)
        res = result.get()
        for data in res:
            for metric in res[data]:
                new_value = np.round(res[data][metric], 4)
                record_to_update = Metrics.query.filter_by(model_id=model_id, data=data, metric=metric).all()[0]
                record_to_update.value = new_value
                db.session.commit()
        return f'Metrics for model {model_id} are updated', 200

    def delete(self, model_id):
        """
        Database DELETE operation.
        Delete performance metrics of the given model from a database.
        """
        Metrics.query.filter_by(model_id=model_id).delete()
        db.session.commit()
        return f'Metrics for model {model_id} are deleted', 200


if __name__ == '__main__':
    db.create_all()
    app.run(host=os.environ['HOST'],
            port=os.environ['PORT'],
            debug=os.environ['DEBUG'])
