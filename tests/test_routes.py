import unittest
from src.app import create_app

class TestRoutes(unittest.TestCase):
    
    def setUp(self):
        # Initialize the app client for testing
        self.app = create_app().test_client()
        self.app.testing = True

    def test_analyze_with_text(self):
        # Define the payload for the API request
        payload = {
            "resume_text": "Python developer with 5 years of experience.",
            "job_description": "Looking for a Python developer."
        }
        
        # Make a POST request to the analyze route
        response = self.app.post("/api/analyze", data=payload)

        # Print the response to the console for debugging
        print("Response Status Code:", response.status_code)
        print("Response JSON:", response.json)
        print("Response Headers:", response.headers)

        # Assert that the response status is OK
        self.assertEqual(response.status_code, 200)

        # Assert that the response content type is JSON
        self.assertEqual(response.content_type, "application/json")

        # Assert that the response contains the expected keys
        self.assertIn("ats_score", response.json)
        self.assertIn("feedback", response.json)

if __name__ == "__main__":
    unittest.main()
