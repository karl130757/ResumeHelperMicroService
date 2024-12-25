
AI-Powered Resume Analyzer MicroService
==========================

This project implements an AI-based resume analysis tool that helps optimize resumes for ATS (Applicant Tracking System) scans. The tool analyzes the resume in relation to a provided job description and provides actionable feedback on how to improve the resume’s alignment with the job description and increase its chances of passing through ATS filters.

Features
--------
- **ATS Compatibility**: The tool analyzes the resume and suggests specific keywords and phrases that should be added to improve compatibility with ATS systems.
  
- **Experience Improvement**: It offers suggestions on how to highlight or improve relevant work experience, including recommendations on action verbs and quantifiable achievements.
  
- **Skills Optimization**: The AI suggests additional technical and soft skills that should be included in the resume based on the job description to match the desired qualifications.
  
- **Structure Enhancement**: The tool provides recommendations on improving the overall structure of the resume, including section order, clarity, layout, and adding additional sections like "Projects" or "Achievements."
  
- **Personalized Feedback**: Based on the job description, the tool gives personalized suggestions to tailor the resume more effectively for the specific role.

- **ATS Score Calculation**: The system computes an ATS compatibility score using spaCy, assessing how well the resume aligns with the job description by analyzing keyword presence, entity recognition, and syntactic structure.

AI Models Used
--------------
The tool uses **GPT-J**, a large-scale language model, and **spaCy**, an open-source library for advanced Natural Language Processing (NLP). GPT-J is used for generating suggestions on how to improve the resume, while spaCy is used to pre-process, analyze text, and calculate the ATS compatibility score by evaluating keyword relevance, skill matching, and job description alignment.

### Model Details
- **GPT-J**
  - **Model Name**: GPT-J
  - **Type**: Transformer-based language model
  - **Version**: GPT-J 6B (6 billion parameters)
  - **Architecture**: GPT (Generative Pretrained Transformer)
  - **Usage**: GPT-J provides specific, actionable feedback for improving resume content, structure, and alignment with job descriptions.

- **spaCy**
  - **Library**: spaCy (used for text processing)
  - **Usage**: spaCy is used to pre-process and analyze the resume content, including named entity recognition (NER), part-of-speech tagging, and syntactic analysis. It plays a key role in computing the ATS score by evaluating the relevance of keywords and the alignment between the resume and job description.

Requirements
------------
To run the project locally, ensure the following are installed:
- Python 3.7+
- Required Python Libraries (listed in `requirements.txt`)
- Access to the GPT-J model (or equivalent transformer model for text generation)
- spaCy (for text analysis and ATS score calculation)


   ```

How It Works
-------------
1. The tool takes a resume and job description as input.
2. The resume is analyzed using **spaCy** for text processing (such as extracting skills, job titles, and experiences) and to calculate the ATS score based on keyword and content relevance.
3. The processed content is fed to **GPT-J**, which compares the resume with the job description and generates feedback based on the analysis.
4. Based on the analysis, the model provides feedback in four key areas:
   - ATS Compatibility
   - Experience
   - Skills
   - Overall Structure
5. The feedback is presented in a structured format for easy integration into the resume.

Future Enhancements
-------------------
- Expand the model to support multiple job categories and industries.
- Include recommendations on resume formatting and visual appeal.
- Enhance feedback quality by training on domain-specific datasets.

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.
