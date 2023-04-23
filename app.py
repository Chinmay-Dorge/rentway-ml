from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the decision tree model
with open('decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv('Dataset.csv')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Create a dataframe from the input data
    print(data)
    input_df = pd.DataFrame.from_dict(data)

    # Select only the columns that were used to train the model
    input_df = input_df[['cloth_type', 'size', 'color', 'fabric', 'brand', 'mrp', 'age_months', 'rental_duration', 'availability', 'condition', 'location', 'occasion']]

    # Make a prediction using the model
    prediction = model.predict(input_df)

    # Return the prediction as a JSON response
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
