import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')
data = data[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]
data['Sex'] = data['Sex'].replace('male', 0)
data['Sex'] = data['Sex'].replace('female', 1)
data.dropna(inplace=True)

X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data[['Survived']]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importance = clf.feature_importances_

print(importance)

