import score
import numpy
import pandas as pd
import os
import requests
import subprocess
import time
import unittest
import joblib
import pytest
import random


@pytest.fixture

# Fixture function to load the best model stored in a joblib file and return it
def model():
        
    model_path = r"best_model.joblib"
    trained_model=joblib.load(model_path)

    return trained_model

# Pytest test case to check if the score function produces valid output without crashing
def test_smoke(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)
    assert label in [0,1]
    assert 0 <= prop <=1

# Checks if the output format of the score function is as expected, ensuring the prediction label is an integer and the propensity score is a float.
def test_format(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)

    assert isinstance(label, int)
    assert isinstance(prop, float)
 
# Verifies that the prediction value returned by the score function is either True or False/ 1 or 0.   
def test_prediction_value(model):
    text = "Sample text"
    threshold = 0.5
    prediction, _ = score.score(text, model, threshold)
    assert prediction in [True, False]

# Validates that the propensity score returned by the score function is between 0 and 1.
def test_propensity_score(model):
    text = "Sample text"
    threshold = 0.5
    _, propensity = score.score(text, model, threshold)
    assert propensity >= 0
    assert propensity <= 1

# Tests the behavior of the score function when the threshold is set to 0, ensuring that the predicted label is 1.
def test_threshold_0(model):
    threshold=0
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==1

# Tests the behavior of the score function when the threshold is set to 1, ensuring that the predicted label is 0.
def test_threshold_1(model):
    threshold=1
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==0

#  Checks if the score function correctly predicts spam for an obvious spam message.
def test_spam(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score("YOU HAVE WON 1 MILLION DOLLARS. SEND YOUR ACCOUNT DETAILS!",model,threshold)
    assert label == 1

# Verifies if the score function correctly predicts non-spam for a legitimate message.
def test_ham(model):
    threshold=1
    sent="Its a real mail, not spam"
    label,prop=score.score("Dogs are better than cats anyday.",model,threshold)
    assert label == 0


# integration test function
def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python app.py')

    # Wait for the app to start up
    time.sleep(15)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)
    
    # Checking if the Flask app is properly configured and running or not.
    assert response.status_code == 200
    assert type(response.text)== str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')

class TestDocker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "assignment_4", "."], check=True)
        # Run the Docker container
        cls.container_id = cls._run_docker_container()
        # Load input texts from the CSV file
        cls.test_df = pd.read_csv("test.csv")
        time.sleep(5)  # Wait for the server to start

    @classmethod
    def _run_docker_container(cls):
        # Run the Docker container and return the container ID
        container_id = subprocess.check_output(["docker", "run", "-d", "-p", "5000:5000", "assignment_4"]).decode("utf-8").strip()
        return container_id

    def test_docker(self):
        # Choose a random text from test data
        random_index = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[random_index, 0]
        # Test the /score endpoint
        response = self._send_request(text)
        self.assertEqual(response.status_code, 200)

    def _send_request(self, text):
        # Send a request to the Docker container's /score endpoint
        test_data = {'text': text}
        response = requests.post('http://127.0.0.1:5000/score', json=test_data)
        return response

    @classmethod
    def tearDownClass(cls):
        # Stop and remove the Docker container
        subprocess.run(["docker", "stop", cls.container_id], check=True)
        subprocess.run(["docker", "rm", cls.container_id], check=True)



if __name__ == '__main__':
    unittest.main()