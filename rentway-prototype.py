import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv('Dataset.csv')
X = df.drop(columns=['rental_price'])
y = df['rental_price']

model = DecisionTreeClassifier()
model.fit(X.values, y.values)

feature_names = ['f_' + str(col) for col in X.columns]
class_names = ['c_' + str(cls) for cls in sorted(y.unique())]

tree.export_graphviz(model, out_file='graph.dot', feature_names=feature_names, class_names=class_names, label='all', rounded=True, filled=True)
