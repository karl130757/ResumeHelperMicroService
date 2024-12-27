import logging
import re
import torch
from typing import Dict, List, Union
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from functools import lru_cache
from src.spacy_model import get_spacy_model  # Assuming this loads a SpaCy model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compile regex patterns for performance
SECTION_REGEX = re.compile(r"(ATS Compatibility|Experience|Skills|Overall Structure):\s*(.*?)\s*(?=\*\*|$)", re.DOTALL)

# Validate inputs
def validate_inputs(resume_text: str, job_description: str):
    if not resume_text.strip() or not job_description.strip():
        raise ValueError("Both resume and job description must be non-empty.")

# Load SpaCy model (cached for performance)
@lru_cache(maxsize=None)
def cached_spacy_model():
    return get_spacy_model()

# Load DistilBERT model and tokenizer (cached for performance)
@lru_cache(maxsize=None)
def cached_distilbert_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

# Function to get DistilBERT embeddings
def get_distilbert_embeddings(text: str) -> torch.Tensor:
    try:
        tokenizer, model = cached_distilbert_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA memory overflow. Switching to CPU processing.")
        torch.cuda.empty_cache()
        return get_distilbert_embeddings(text)
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise

# Function to extract named entities from text using SpaCy
def extract_named_entities(text: str) -> List[Dict[str, str]]:
    spacy_model = cached_spacy_model()
    doc = spacy_model(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

# Function to extract keywords (nouns and proper nouns) using SpaCy
def analyze_keywords_with_spacy(text: str) -> Dict[str, Union[List, Counter]]:
    spacy_model = cached_spacy_model()
    doc = spacy_model(text)

    # Extract keywords (nouns/proper nouns) and filter stop words
    keywords = [
        token.lemma_ for token in doc
        if token.is_alpha and token.pos_ in {"NOUN", "PROPN"} and token.lemma_.lower() not in STOP_WORDS
    ]
    entities = extract_named_entities(text)

    return {
        "keywords": Counter(keywords).most_common(10),
        "entities": entities,
    }

# Function to calculate ATS score based on semantic similarity using DistilBERT
def calculate_ats_score(resume_text: str, job_desc_text: str) -> float:
    try:
        # Get embeddings for both resume and job description using DistilBERT
        resume_embedding = get_distilbert_embeddings(resume_text)
        job_desc_embedding = get_distilbert_embeddings(job_desc_text)

        # Compute cosine similarity between the embeddings
        similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_desc_embedding.detach().numpy())[0][0]
        return similarity_score
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# Function to extract key skills from job description using SpaCy
def extract_job_skills(job_description: str) -> List[str]:
    # Analyze keywords and extract skills that are nouns (proper and common)
    job_desc_analysis = analyze_keywords_with_spacy(job_description)
    job_desc_keywords = [word for word, _ in job_desc_analysis["keywords"]]
    
    return job_desc_keywords

# Function to analyze and give feedback on the overall structure of the resume
def analyze_resume_structure(resume_text: str, job_description: str) -> str:
    # Essential sections we expect to find in the resume
    required_sections = ["experience", "skills", "education", "certifications", "projects"]

    # Check if the resume contains these sections
    missing_sections = [section for section in required_sections if section not in resume_text.lower()]

    # Feedback based on the presence of required sections
    if missing_sections:
        missing_feedback = f"Your resume is missing the following key sections: {', '.join(missing_sections)}. Make sure to include these sections to enhance the structure of your resume."
    else:
        missing_feedback = "Your resume includes all the essential sections."

    # Analyze the quality of the 'Experience' section and other important sections using DistilBERT
    experience_section_feedback = analyze_section_quality(resume_text, job_description, "experience")
    skills_section_feedback = analyze_section_quality(resume_text, job_description, "skills")

    # Combine all structure feedback
    return f"{missing_feedback} {experience_section_feedback} {skills_section_feedback}"

# Function to analyze the quality of a specific section using DistilBERT (e.g., experience, skills)
def analyze_section_quality(resume_text: str, job_description: str, section: str) -> str:
    try:
        # Extract the relevant part of the resume for the section
        section_text = extract_section_text(resume_text, section)

        # Check if the section contains job-specific skills or relevant information
        job_skills = extract_job_skills(job_description)
        section_keywords = analyze_keywords_with_spacy(section_text)["keywords"]
        section_keywords = [word for word, _ in section_keywords]

        # Compute similarity of section content with job description using DistilBERT
        section_embedding = get_distilbert_embeddings(section_text)
        job_desc_embedding = get_distilbert_embeddings(job_description)
        similarity_score = cosine_similarity(section_embedding.detach().numpy(), job_desc_embedding.detach().numpy())[0][0]

        # Provide feedback based on semantic similarity
        if similarity_score < 0.5:  # Example threshold for low similarity
            feedback = f"Your {section} section could be improved. It doesn't strongly align with the job description. Try including more relevant details and keywords, such as {', '.join(job_skills)}."
        else:
            feedback = f"Your {section} section aligns well with the job description."

        return feedback
    except Exception as e:
        logging.error(f"Error analyzing section '{section}': {str(e)}")
        return f"Unable to analyze the {section} section due to an error."

# Function to extract the section text from the resume
def extract_section_text(resume_text: str, section: str) -> str:
    # Define regex patterns for extracting relevant sections (like Experience, Skills, etc.)
    section_patterns = {
        "experience": r"(Experience|Work History|Employment|Professional Experience):\s*(.*?)(?=\n|$)",
        "skills": r"(Skills|Technical Skills|Core Competencies):\s*(.*?)(?=\n|$)",
        "education": r"(Education):\s*(.*?)(?=\n|$)",
        "certifications": r"(Certifications):\s*(.*?)(?=\n|$)",
        "projects": r"(Projects):\s*(.*?)(?=\n|$)"
    }
    
    pattern = section_patterns.get(section.lower())
    if pattern:
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            return match.group(2).strip()
    return ""  # Return empty if section is not found

# Function to generate feedback based on ATS score and keywords using DistilBERT and SpaCy
def generate_feedback(resume_text: str, job_description: str) -> Dict[str, str]:
    try:
        similarity_score = calculate_ats_score(resume_text, job_description)
        ats_feedback = f"Your resume's alignment with the job description is {similarity_score * 100:.2f}%. To improve, consider adding more keywords related to the job description."

        # Extract job-specific skills from job description
        job_skills = extract_job_skills(job_description)

        # Analyze keywords using SpaCy
        resume_analysis = analyze_keywords_with_spacy(resume_text)
        missing_keywords = list(set(job_skills) - set([word for word, _ in resume_analysis["keywords"]]))

        skills_feedback = (
            f"Consider emphasizing these missing skills/experiences: {', '.join(missing_keywords)}."
            if missing_keywords else "Your resume matches most of the required skills and experiences."
        )

        # Structural Feedback using DistilBERT (analyzing sections)
        structure_feedback = analyze_resume_structure(resume_text, job_description)

        # Soft Skills Feedback
        soft_skills_feedback = (
            "Your resume doesn't mention soft skills such as 'teamwork' or 'communication'. These are highly valued for this role. Consider adding examples of when you've worked in teams or communicated complex information effectively."
            if not any(skill in resume_text.lower() for skill in ["communication", "teamwork", "collaboration"])
            else "Great job! You've included soft skills like 'teamwork' and 'communication'."
        )

        return {
            "ATS Compatibility": ats_feedback,
            "Experience and Skills": skills_feedback,
            "Suggested Keywords": f"Suggested keywords to add: {', '.join(missing_keywords)}. You can include them in your skills section or mention them within your experience.",
            "Overall Structure": structure_feedback,
            "Soft Skills": soft_skills_feedback
        }

    except Exception as e:
        logging.error(f"Error generating feedback: {str(e)}")
        raise

# Function to analyze resume and generate feedback
def analyze_resume(resume_text: str, job_description: str) -> Dict:
    try:
        validate_inputs(resume_text, job_description)

        # Analyze with SpaCy for keyword extraction and named entities
        resume_analysis = analyze_keywords_with_spacy(resume_text)
        job_desc_analysis = analyze_keywords_with_spacy(job_description)

        # Calculate ATS score using DistilBERT similarity
        ats_score = calculate_ats_score(resume_text, job_description)

        # Generate feedback based on the analysis
        feedback = generate_feedback(resume_text, job_description)

        return {
            "word_count": len(resume_text.split()),
            "sentence_count": len(re.findall(r'\.', resume_text)),
            "top_keywords": [{"word": word, "count": count} for word, count in resume_analysis["keywords"]],
            "named_entities": resume_analysis["entities"],
            "ats_score": round(float(ats_score) * 100),  # Round to whole number
            "feedback": feedback,
        }

    except Exception as e:
        logging.error(f"Error during resume analysis: {str(e)}")
        return {"error": str(e)}
