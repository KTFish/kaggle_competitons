# This file contains the data cleaning process
# The analysis why it should be done so is in the Jupyter Notebook script

import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
train = pd.read_csv('./data/train.csv')
y_train = train['Survived']
train.drop(['Survived'], axis=1, inplace=True)
test = pd.read_csv('./data/test.csv')


def refactor(df):
    """Huge function for cleaning both datasets."""

    # Fare ---> FareCategory
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)  # Test set has one Nan value

    kmeans = KMeans(n_clusters=5)
    fare = df['Fare'].to_numpy().reshape(-1, 1)
    kmeans.fit(fare)

    df['FareCategory'] = kmeans.labels_
    df.drop(['Fare'], axis=1, inplace=True)

    # SibSp + Parch ---> FamillySize
    df['FamillySize'] = df['Parch'] + df['SibSp']

    # Name ---> Title
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.')
    df['Title'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col',
                         'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
                        ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr',
                         'Other'],
                        inplace=True)

    # Age
    mean_age = train[['Title', 'Age']].groupby(['Title'], as_index=False).mean().sort_values(by='Age')
    mean_for_title = lambda title: mean_age[mean_age['Title'] == title].Age.values[0]
    df['Age'].fillna(-1, inplace=True)
    for title in train['Title'].unique():
        df.loc[(df['Age'] == -1) & (df['Title'] == title), 'Age'] = mean_for_title(title)

    # Age ---> Category
    df.loc[df['Age'] <= 11, 'Age'] = 0
    df.loc[(df['Age'] > 11) & (df['Age'] <= 18), 'Age'] = 1
    df.loc[(df['Age'] > 18) & (df['Age'] <= 22), 'Age'] = 2
    df.loc[(df['Age'] > 22) & (df['Age'] <= 27), 'Age'] = 3
    df.loc[(df['Age'] > 27) & (df['Age'] <= 30), 'Age'] = 4
    df.loc[(df['Age'] > 30) & (df['Age'] <= 40), 'Age'] = 5
    df.loc[(df['Age'] > 40) & (df['Age'] <= 65), 'Age'] = 6
    df.loc[df['Age'] > 65, 'Age'] = 7

    df['Age'] = df['Age'].astype('int64')

    # Sex
    df['IsMale'] = df['Sex'] == 'male'
    train['IsMale'] = train['IsMale'].astype('int64')

    # Embarked
    df['Embarked'].fillna(df['Embarked'].mode()[0])

    # drop columns
    drop_cols = ['Cabin', 'SibSp', 'Parch', 'Ticket', 'Name', 'Sex', 'PassengerId']
    df = df.drop(drop_cols, axis=1)

    # encode
    df = pd.get_dummies(df, columns=['Pclass', 'Age', 'Embarked', 'Title', 'FamillySize', 'FareCategory'],
                        prefix=['Pclass', 'Age', 'Embarked', 'Title', 'FamillySize', 'FareCategory'])
    return df


X_train = refactor(train)
X_test = refactor(test)
X_train.to_csv('./data/train2.csv')
X_test.to_csv('./data/test2.csv')

print(X_test.info())