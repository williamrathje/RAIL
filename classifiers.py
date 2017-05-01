# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# classifiers
from sklearn import linear_model, decomposition, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# tools
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# eval
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


### Pick a model. Logistic, randomforests, support vector machines, and naive bayes seem good...

model = linear_model.LogisticRegression()
#model = RandomForestClassifier()
#model = svm.SVC(gamma=0.001, C=100.)
#model = GaussianNB()

# Use PCA for feature selection
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('model', model)])

# Use significant chi2 for feature selection
#selector = SelectKBest(chi2, k=10)
#pipe = Pipeline(steps=[('feature_selection', selector), ('model', model)])

# Load up the dataset
cancer = datasets.load_breast_cancer()
X_data = cancer.data
y_target = cancer.target

# Search for the best number of features with grid search

#n_components = [n for n in range(1, 30)]       # try all possibilities
n_components = [5, 10, 20, 30]      # try a few good possibilities
model = GridSearchCV(pipe, dict(pca__n_components=n_components))        # pca
#model = GridSearchCV(pipe, dict(feature_selection__k=n_components))    # chi2

# evaluate pipeline with 5-fold cross validation
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
results = cross_val_score(model, X_data, y_target, cv=kfold)
print(results.mean())