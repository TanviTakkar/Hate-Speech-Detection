from fastapi.testclient import TestClient
from hate_speech_api import app  # Replace 'your_module' with your actual module name

client = TestClient(app)

def test_classifier():
    response = client.post("/classifier/", json={"text": "Please shut the door"})
    assert response.status_code == 200
    data = response.json()
    assert "Label" in data
    assert "score" in data
