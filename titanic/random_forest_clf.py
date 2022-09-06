import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv('./data/train2.csv').drop('Unnamed: 0', axis=1)
X_test = pd.read_csv('./data/test2.csv').drop('Unnamed: 0', axis=1)
Id_test = pd.read_csv('./data/test.csv', usecols=['PassengerId'])
y_train = pd.read_csv('./data/train.csv', usecols=['Survived']).to_numpy().ravel()

# Some parameters to test.

# Number of trees in the forest
n_estimators = [20, 30, 40, 100, 400, 700]

# Maximum number of levels in the tree
max_depth = [2, 3, 4, 7, 10, 15]

# Number of features to consider at every split
max_features = ['sqrt', 'auto']

# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 7]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]

# Method of selecting samples for training each tree
bootstrap = [True, False]

param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    #     'max_features': max_features,
    #     'max_depth': max_depth,
    #     'min_samples_split': min_samples_split,
    #     'min_samples_leaf': min_samples_leaf,
    #     'bootstrap': bootstrap
}

forest_clf = RandomForestClassifier(random_state=2002)

forest_grid = GridSearchCV(estimator=forest_clf,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3, verbose=2, n_jobs=-1)

forest_grid.fit(X_train, y_train)
print(forest_grid.best_score_)
print(forest_grid.best_estimator_)
print(forest_grid.best_params_)

# Best accuracy for now: 81,7 %
# RandomForestClassifier(max_depth=4, n_estimators=700, random_state=2002)
