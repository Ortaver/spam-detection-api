import requests

url = "https://spam-detection-api-yt3f.onrender.com/predict"

data = {
    "message": "Free money! Click now!"
}

response = requests.post(url, json=data)

print(response.json())