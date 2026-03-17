import requests

url = "http://127.0.0.1:5000/predict"
data = {"message": "You have been selected for a free vacation!"}

response = requests.post(url, json=data)
print("Prediction:", response.json())
