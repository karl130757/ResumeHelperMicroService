
# ResumeHelperMicroService

## Overview
**ResumeHelperMicroService** is an AI-powered tool designed to help job candidates optimize their resumes for Applicant Tracking Systems (ATS). It analyzes the resume and job description and provides actionable feedback to improve ATS compatibility. The service uses SpaCy for NLP tasks and calculates an ATS score based on the input.

## Features
- **ATS Compatibility Feedback**: Recommends keywords and phrases to improve ATS compatibility.
- **Experience Section Feedback**: Suggests how to enhance work experience.
- **Skills Feedback**: Identifies missing or relevant skills to include.
- **Overall Structure Feedback**: Provides tips on improving resume layout and structure.
- **Spacy-based ATS Score Calculation**: Calculates an ATS score to assess how well the resume matches the job description.

## Technologies Used
- **Flask**: Backend framework for the web service.
- **SpaCy**: NLP library used for resume analysis and calculating the ATS score.
- **Python**: Backend logic for integrating SpaCy with custom analysis.
- **AI Models**: Pre-trained SpaCy models for text analysis (tokenization, entity recognition, etc.).

## How It Works
1. **Input**: Users can upload a resume file (e.g., `.txt`, `.pdf`, `.docx`) and a job description (either text or file upload).
2. **Processing**: The resume and job description are processed using SpaCy for text analysis, followed by ATS score calculation.
3. **Feedback**: The system provides feedback on ATS compatibility, work experience, skills, structure, and more.
4. **Output**: Actionable recommendations are returned to improve the resume.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/karl130757/ResumeHelperMicroService.git
    cd ResumeHelperMicroService
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Install SpaCy and the necessary language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```
4. Run the Flask application:
    ```bash
    python app.py
    ```
5. Access the service locally at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
To use the **ResumeHelperMicroService**, send a POST request to the `/analyze` endpoint. Users can either upload files or provide the resume and job description as text in the request.

### Example Request (Text):
```json
POST /analyze
{
   "resume": "Python developer with 5 years of experience...",
   "job_description": "Looking for a Python developer..."
}
```

### Example Request (File Upload):
```html
<form action="/analyze" method="post" enctype="multipart/form-data">
   <input type="file" name="resume">
   <input type="file" name="job_description">
   <button type="submit">Analyze</button>
</form>
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About
No description, website, or topics provided.
