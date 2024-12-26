
# ResumeHelperMicroService

## Overview
**ResumeHelperMicroService** is an AI-powered tool designed to help job candidates optimize their resumes for Applicant Tracking Systems (ATS). It analyzes resumes and job descriptions, providing actionable feedback to improve ATS compatibility. The service uses SpaCy and DistilBERT for natural language processing and semantic analysis to generate feedback.

## Features:
- **ATS Compatibility**: Analyzes the alignment between the resume and the job description and suggests keywords and phrases for improved ATS compatibility.
- **Experience Enhancement**: Offers suggestions to improve work experience sections with quantifiable achievements and impactful language.
- **Skills Matching**: Identifies technical and soft skills that align with the job description to ensure a better match.
- **Resume Structure Optimization**: Recommends improvements in structure, organization, and clarity.
- **Actionable Feedback**: Provides concise feedback on how to refine each section of the resume.
- **Named Entity Recognition (NER)**: Extracts named entities such as skills, tools, and experience durations to enhance the analysis.
- **ATS Score Calculation**: Computes an ATS compatibility score based on the semantic similarity between the resume and the job description using SpaCy and DistilBERT.

## AI Models Used:
- **DistilBERT**: Employed for semantic similarity analysis and generating feedback for better alignment between resumes and job descriptions.
- **spaCy**: Used for natural language processing tasks such as named entity recognition and ATS score computation.

## Technologies Used
- **Flask**: Backend framework for the web service.
- **SpaCy**: NLP library used for resume analysis and ATS scoring.
- **DistilBERT**: Transformer model for semantic similarity and analysis.
- **Python**: Backend logic and integration.

## How It Works
1. **Input**: Users upload a resume file (e.g., `.txt`, `.pdf`, `.docx`) and a job description (text or file upload).
2. **Processing**: The system processes the input with SpaCy and DistilBERT to calculate an ATS score and generate feedback.
3. **Feedback**: Actionable recommendations are provided to improve the resume.
4. **Output**: The system returns feedback on ATS compatibility, work experience, skills, structure, and more.

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
To use the **ResumeHelperMicroService**, send a POST request to the `/analyze` endpoint. Users can upload files or provide the resume and job description as text in the request.

### Example Request (Text):
POST /analyze
```json
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
