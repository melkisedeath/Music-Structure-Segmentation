import numpy as np

from metric_learn import NCA, LMNN, LFDA, MLKR
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class ClfSwitcher(BaseEstimator):

    def __init__(self, 
        estimator = NCA(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)



# Create a Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ClfSwitcher()),
])


parameters = [
    {
        'clf__estimator': [NCA()], # SVM if hinge loss / logreg if log loss
        # 'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
        'clf__estimator__max_iter': [1000],
        # 'clf__estimator__tol': [1e-4],
        # 'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
    },
    {
        'clf__estimator': [LMNN()],  
        'clf__estimator__k' : (5),
        'clf__estimator__learn_rate' : [1e-6], 
    },
    {
        'clf__estimator': [LFDA()],
        'clf__estimator__k' : (2),
        'clf__estimator__dim' : (2),   
    },
    {
        'clf__estimator': [MLKR()],
        'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
    },
]

gscv = GridSearchCV(pipeline, parameters, cv=5, n_jobs=12, return_train_score=False, verbose=3)
gscv.fit(train_data, train_labels)