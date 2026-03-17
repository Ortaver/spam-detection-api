document.getElementById('spamForm').addEventListener('submit', function(e) {
  e.preventDefault();
  const message = document.getElementById('message').value;

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message: message })
  })
  .then(response => response.json())
  .then(data => {
    const resultDiv = document.getElementById('result');
    if (data.error) {
      resultDiv.innerText = 'Error: ' + data.error;
      resultDiv.className = '';
    } else {
      const labelClass = data.prediction === 'spam' ? 'spam' : 'ham';
      resultDiv.innerHTML = `Prediction: <span class="${labelClass}">${data.prediction.toUpperCase()}</span><br>Confidence: ${data.confidence}`;
    }
  })
  .catch(error => {
    const resultDiv = document.getElementById('result');
    resultDiv.innerText = 'Error: ' + error.message;
    resultDiv.className = '';
  });
});
