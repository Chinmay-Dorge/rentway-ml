
import pandas as pd
df = pd.read_csv('Dataset.csv')
df

X = df.drop(columns=['rental_price'])
X

y = df['rental_price']
y

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
# Sending the data so we can use it to predict results
# Use .values to avoid warning
model.fit(X.values, y.values)

predictions = model.predict([[2,6,4,2,34,25999,3,4,5,2,4,2]])
predictions


