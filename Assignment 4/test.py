import os
import requests
import time
import pickle
import pytest
import score
import joblib 

# Define the model and test data
model = joblib.load("/mnt/d/Applied ML/Assignment 4/Best_LR.pkl")
text = "Television is the opium of the masses."
obv_ham_text = "The Sun rises from the East."
obv_spam_text = "Press this link to win an aeroplane for free."
threshold = 0.55

# Define tests using pytest-style functions and fixtures
def test_smoke():
    label, prop = score.score(text, model, threshold)
    assert label is not None
    assert prop is not None

def test_input_formats():
    label, prop = score.score(text, model, threshold)
    assert isinstance(text, str)
    assert isinstance(threshold, float)
    assert isinstance(label, bool)
    assert isinstance(prop, float)

def test_pred_value():
    label, _ = score.score(obv_ham_text, model, threshold)
    assert label in [True, False]

def test_prop_value():
    _, prop = score.score(text, model, threshold)
    assert 0 <= prop <= 1

def test_pred_thres_0():
    label, _ = score.score(text, model, threshold=1)
    assert label is True

def test_pred_thres_1():
    label, _ = score.score(text, model, threshold=0)
    assert label is False

def test_obvious_spam():
    label, _ = score.score(obv_spam_text, model, threshold)
    assert label is True

def test_obvious_ham():
    label, _ = score.score(obv_ham_text, model, threshold)
    assert label is False

@pytest.fixture
def run_container():
    # Build the Docker image
    os.system("docker build -t flaskapp .")

    # Run the Docker container in detached mode
    container_id = os.popen("docker run -d -p 5000:5000 flaskapp").read().strip()

    # Wait for the app to start up
    time.sleep(4)

    yield container_id

    # Stop and remove the Docker container
    os.system(f"docker stop {container_id}")
    os.system(f"docker rm {container_id}")

def test_docker(run_container):
    container_id = run_container

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    assert response.status_code == 200
