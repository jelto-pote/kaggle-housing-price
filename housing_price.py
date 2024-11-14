from flask import Flask, request, jsonify
from joblib import load
from numpy import nan
from pandas import DataFrame
from os import environ

# Initialize the Flask application
app = Flask(__name__)

# Load the model
model = load('models/CatBoostRegressor_current.joblib')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()


        features = data['features']

        # Convert features into a DataFrame
        df = DataFrame([features], columns=['longitude', 'latitude', 'housing_median_age', 
                                           'total_rooms', 'total_bedrooms', 'population', 
                                           'households', 'median_income', 'ocean_proximity'])

        df.drop(['longitude', 'latitude'], axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..

        df = add_interaction_feature_number(df, 'median_income', 'households', '/')
        df = add_interaction_feature_number(df, 'population', 'households', '/')

        print(df.head())

        # Predict using the loaded model
        prediction = model.predict(df)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def add_interaction_feature_number(df, col1, col2, operation, drop1=False, drop2=False):
    if operation == '+':
        feature = col1 + '_' + col2 + '_sum'
        df[feature] = df[col1] + df[col2]
    elif operation == '-':
        feature = col1 + '_' + col2 + '_diff'
        df[feature] = df[col1] - df[col2]
    elif operation == '*':
        feature = col1 + '_' + col2 + '_product'
        df[feature] = df[col1] * df[col2]
    elif operation == '/':
        feature = col1 + '_' + col2 + '_division'
        # Prevent division by zero
        df[feature] = df[col1] / df[col2].replace(0, nan)
    
    # Drop original columns if requested
    if drop1:
        df.drop(col1, axis=1, inplace=True)
    if drop2:
        df.drop(col2, axis=1, inplace=True)

    return df


# Ensure the app binds to the correct port (Heroku will assign a PORT environment variable)
if __name__ == '__main__':
    port = int(environ.get('PORT', 5000))  # Use PORT from the environment, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)  # Bind to 0.0.0.0 for accessibility from outside
