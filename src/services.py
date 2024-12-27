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
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compile regex patterns for performance
SECTION_PATTERNS = {
    "experience": r"(Experience|Work History|Employment|Professional Experience):\s*(.*?)(?=\n|$)",
    "skills": r"(Skills|Technical Skills|Core Competencies):\s*(.*?)(?=\n|$)",
    "education": r"(Education):\s*(.*?)(?=\n|$)",
    "certifications": r"(Certifications):\s*(.*?)(?=\n|$)",
    "projects": r"(Projects):\s*(.*?)(?=\n|$)"
}

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
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        return torch.zeros(1, 768)  # Return a zero tensor in case of an error

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

# Function to calculate ATS score based on semantic similarity using DistilBERT and keyword matching
def calculate_ats_score(resume_text: str, job_desc_text: str, job_skills: List[str]) -> float:
    try:
        # Ensure job_skills is a list of strings (not tuples)
        if isinstance(job_skills[0], tuple):
            job_skills = [skill for skill, _ in job_skills]  # Extract just the skills from tuples

        # Get embeddings for both resume and job description using DistilBERT
        resume_embedding = get_distilbert_embeddings(resume_text)
        job_desc_embedding = get_distilbert_embeddings(job_desc_text)

        # Compute cosine similarity between the embeddings
        similarity_score = cosine_similarity(resume_embedding.detach().numpy(), job_desc_embedding.detach().numpy())[0][0]

        # Add keyword match score to adjust ATS compatibility based on domain relevance
        job_skills_in_resume = sum([1 for skill in job_skills if skill.lower() in resume_text.lower()])
        job_skills_in_desc = len(job_skills)
        skills_match_score = job_skills_in_resume / job_skills_in_desc if job_skills_in_desc else 0
        
        # Combine semantic similarity and keyword match
        return (similarity_score + skills_match_score) / 2
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# Function to extract job skills from job description using SpaCy
def extract_job_skills(job_description: str) -> List[str]:
    # Analyze keywords and extract skills that are nouns (proper and common)
    job_desc_analysis = analyze_keywords_with_spacy(job_description)
    job_desc_keywords = [word for word, _ in job_desc_analysis["keywords"]]  # Only the words, not the counts
    return job_desc_keywords

# Function to extract the section text from the resume
def extract_section_text(resume_text: str, section: str) -> str:
    pattern = SECTION_PATTERNS.get(section.lower())
    if pattern:
        match = re.search(pattern, resume_text, re.DOTALL)
        if match:
            return match.group(2).strip()
    return ""  # Return empty if section is not found

# Function to analyze the quality of a specific section using DistilBERT
def analyze_section_quality(resume_text: str, job_description: str, section: str) -> str:
    section_text = extract_section_text(resume_text, section)
    if not section_text:
        return f"Your {section} section is missing or incomplete."

    job_skills = extract_job_skills(job_description)
    section_keywords = analyze_keywords_with_spacy(section_text)["keywords"]
    section_keywords = [word for word, _ in section_keywords]

    # Compute similarity of section content with job description using DistilBERT
    section_embedding = get_distilbert_embeddings(section_text)
    job_desc_embedding = get_distilbert_embeddings(job_description)
    similarity_score = cosine_similarity(section_embedding.detach().numpy(), job_desc_embedding.detach().numpy())[0][0]

    if similarity_score < 0.5:
        feedback = f"Your {section} section could be improved. It doesn't strongly align with the job description. Try including more relevant details and keywords, such as {', '.join(job_skills)}."
    else:
        feedback = f"Your {section} section aligns well with the job description."

    return feedback

# Function to analyze resume structure (missing sections, etc.)
def analyze_resume_structure(resume_text: str, job_description: str) -> str:
    # Essential sections we expect to find in the resume
    required_sections = ["experience", "skills", "education", "certifications", "projects"]
    missing_sections = [section for section in required_sections if section not in resume_text.lower()]

    if missing_sections:
        missing_feedback = f"Your resume is missing the following key sections: {', '.join(missing_sections)}. Consider adding them."
    else:
        missing_feedback = "Your resume includes all the essential sections."

    # Use ThreadPoolExecutor for parallel section analysis
    with ThreadPoolExecutor() as executor:
        future_experience = executor.submit(analyze_section_quality, resume_text, job_description, "experience")
        future_skills = executor.submit(analyze_section_quality, resume_text, job_description, "skills")
        
        experience_feedback = future_experience.result()
        skills_feedback = future_skills.result()

    return f"{missing_feedback} {experience_feedback} {skills_feedback}"

# Function to generate feedback based on ATS score and keywords using DistilBERT and SpaCy
def generate_feedback(resume_text: str, job_description: str) -> Dict[str, str]:
    try:
        job_skills = extract_job_skills(job_description)
        
        # Calculate ATS score with keyword relevance
        ats_score = calculate_ats_score(resume_text, job_description, job_skills)
        ats_feedback = f"Your resume's alignment with the job description is {ats_score * 100:.2f}%. To improve, consider adding more keywords related to the job description."

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
        ats_score = calculate_ats_score(resume_text, job_description, job_desc_analysis["keywords"])

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
