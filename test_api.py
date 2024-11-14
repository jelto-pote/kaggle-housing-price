import requests

# Example input data
data = {
    'features': [-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 'NEAR BAY']
}

# Send a POST request to your local Flask API
url = 'https://kaggle-housing-price-0a8ae3c117ac.herokuapp.com/predict'
response = requests.post(url, json=data)

# Check the response status and print the result
if response.status_code == 200:
    print("Prediction Response:", response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
