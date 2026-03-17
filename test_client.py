import requests

def test_spam_prediction():
    url = "http://127.0.0.1:5000/predict"  # Your Flask API URL
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hello, just checking in on our meeting schedule.",
        "You have been selected for a $1000 cash prize!",
        "Can you send me the report by 5 PM?",
    ]

    for msg in test_messages:
        payload = {"message": msg}
        response = requests.post(url, json=payload)
        print(f"\nInput: {msg}\nPrediction: {response.json()['prediction']}")

if __name__ == "__main__":
    test_spam_prediction()
