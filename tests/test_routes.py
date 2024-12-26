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
            "resume_text": (
                "Python developer with 5 years of experience in designing, "
                "developing, and maintaining scalable applications. Proficient in frameworks such as Django, Flask, and FastAPI. "
                "Skilled in integrating APIs, working with relational and non-relational databases like PostgreSQL and MongoDB, and "
                "deploying applications using Docker and Kubernetes. Strong knowledge of cloud services such as AWS and GCP. "
                "Proven experience in collaborating with cross-functional teams to deliver high-quality software solutions."
            ),
            "job_description": (
                "Looking for a skilled Python developer to join our dynamic team. The ideal candidate should have a minimum of 3-5 years of experience "
                "working with Python frameworks such as Django or Flask. Responsibilities include developing RESTful APIs, integrating with third-party services, "
                "managing databases, and deploying scalable applications. Experience with cloud platforms, CI/CD pipelines, and containerization tools like Docker is a plus. "
                "Excellent problem-solving skills and the ability to work collaboratively in an agile environment are required."
            )
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
