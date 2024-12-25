
# ResumeHelperMicroService

## Overview
ResumeHelperMicroService is an AI-powered resume analysis tool designed to help candidates improve their resumes to pass through Applicant Tracking Systems (ATS). By leveraging advanced natural language processing (NLP) techniques, the service provides actionable feedback to optimize resumes for better ATS compatibility. This includes suggestions for keywords, experience, skills, structure, and more.

## Features
- **ATS Compatibility Feedback**: The tool provides recommendations on keywords and phrases that should be added to the resume for better ATS performance.
- **Experience Section Feedback**: Suggestions on how to improve or highlight relevant work experience for the position.
- **Skills Feedback**: A list of recommended skills to include based on the job description.
- **Overall Structure Feedback**: Insights on how to improve the layout and organization of the resume for better clarity and impact.
- **Spacy-based ATS Score Calculation**: Utilizes SpaCy to compute the ATS compatibility score based on resume content and job description. This allows the system to quantify how well the resume matches the job's requirements.

## Technologies Used
- **Flask**: Flask is used to build the backend API that processes resume data and job descriptions.
- **SpaCy**: SpaCy, a powerful NLP library, is used to compute the ATS score and provide suggestions for text improvement.
- **Python**: The core logic of the microservice is implemented in Python, combining SpaCy's NLP capabilities with custom logic for analyzing resumes and job descriptions.
- **AI Models**: Pre-trained models from SpaCy are used for language processing tasks like tokenization, part-of-speech tagging, and named entity recognition to evaluate resume content.

## How It Works
1. **Input**: Users provide their resume and the job description in a structured format (e.g., text or JSON).
2. **Processing**: The resume is parsed, and SpaCy is used to extract key information, analyze the structure, and compute an ATS score.
3. **Feedback**: The system generates suggestions for improving the resume in specific areas like ATS compatibility, experience, skills, and overall structure.
4. **Output**: Feedback is returned to the user in a structured format, providing clear instructions for improvement.

## Installation

To get started with the ResumeHelperMicroService, clone the repository and set up the environment:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ResumeHelperMicroService.git
   ```

2. Install dependencies:
   ```
   cd ResumeHelperMicroService
   pip install -r requirements.txt
   ```

3. Install SpaCy and download the language model:
   ```
   python -m spacy download en_core_web_sm
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Access the service locally at:
   ```
   http://127.0.0.1:5000
   ```

## Usage

To use the ResumeHelperMicroService, send a POST request to the `/analyze` endpoint with the resume and job description as parameters. The API will return feedback for the specified sections.

Example:
```
POST /analyze
{
   "resume": "Python developer with 5 years of experience...",
   "job_description": "Looking for a Python developer..."
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
