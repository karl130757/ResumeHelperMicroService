import logging
from typing import Dict, List
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.spacy_model import get_spacy_model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from spacy.lang.en.stop_words import STOP_WORDS

# Compile regex patterns for performance
SECTION_REGEX = re.compile(r"(ATS Compatibility|Experience|Skills|Overall Structure):\s*(.*?)\s*(?=\*\*|$)", re.DOTALL)

@lru_cache(maxsize=None)
def cached_spacy_model():
    """Cache the SpaCy model for faster reuse."""
    return get_spacy_model()

# Cache DistilBERT tokenizer and model for efficiency
@lru_cache(maxsize=None)
def cached_distilbert_model():
    """Cache the DistilBERT model and tokenizer for faster reuse."""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

def get_embeddings(text: str) -> torch.Tensor:
    """Generate embeddings using DistilBERT."""
    tokenizer, model = cached_distilbert_model()  # Use cached model and tokenizer
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings

def calculate_ats_score(resume_keywords: List[str], job_desc_keywords: List[str]) -> float:
    """
    Calculate a normalized ATS score based on the number of relevant keywords in both the resume and the job description.
    """
    common_keywords = set(resume_keywords) & set(job_desc_keywords)
    total_keywords = set(job_desc_keywords)
    
    if total_keywords:
        return (len(common_keywords) / len(total_keywords)) * 100  # Percentage of common keywords
    else:
        return 0  # No keywords to match

def generate_feedback(resume_text: str, job_description: str) -> str:
    """
    Analyze resume and job description using DistilBERT and generate semantic similarity-based feedback.
    """
    # Get embeddings for both the resume and job description
    resume_embedding = get_embeddings(resume_text)
    job_desc_embedding = get_embeddings(job_description)

    # Calculate cosine similarity between the resume and job description embeddings
    similarity_score = cosine_similarity(resume_embedding, job_desc_embedding)[0][0]

    ats_feedback = f"ATS Compatibility:\nThe resume's alignment with the job description is {similarity_score * 100:.2f}% based on semantic similarity."

    # Dynamic suggestions based on job description and resume content

    # Extract key experience and skills from both resume and job description
    resume_experience = analyze_keywords_with_spacy(resume_text)
    job_desc_experience = analyze_keywords_with_spacy(job_description)

    # Experience Feedback: Look for missing key experience areas
    missing_experience = [
        skill for skill, _ in job_desc_experience["keywords"]
        if skill not in [word for word, _ in resume_experience["keywords"]]
    ]
    if missing_experience:
        experience_feedback = f"Experience: The resume mentions some key technologies, but make sure to emphasize experience with the following: {', '.join(missing_experience)}. Consider adding relevant projects or achievements to demonstrate your expertise in these areas."
    else:
        experience_feedback = "Experience: The resume aligns well with the experience required for the job. Consider highlighting specific projects or responsibilities that showcase your experience with the listed technologies."

    # Skills Feedback: Compare resume skills to job description
    missing_skills = [
        skill for skill, _ in job_desc_experience["keywords"]
        if skill not in [word for word, _ in resume_experience["keywords"]]
    ]
    if missing_skills:
        skills_feedback = f"Skills: Consider adding specific experience with the following skills to match the job description: {', '.join(missing_skills)}. Include any specific tools or technologies you have worked with."
    else:
        skills_feedback = "Skills: The resume lists many of the relevant skills required for the job. Consider adding further details, such as specific tools or technologies used in your projects."

    # Structure Feedback: Look for structure gaps (e.g., missing certifications, education, etc.)
    if "certifications" not in resume_text.lower():
        structure_feedback = "Overall Structure: The resume could benefit from a section highlighting relevant certifications or training courses."
    elif "education" not in resume_text.lower():
        structure_feedback = "Overall Structure: Ensure that the education section is clearly outlined if it is relevant to the job."
    else:
        structure_feedback = "Overall Structure: The resume is well-structured, but ensuring that relevant certifications or training courses are emphasized can further enhance its impact."

    return f"""
    **ATS Compatibility:**
    {ats_feedback}

    **Experience:**
    {experience_feedback}

    **Skills:**
    {skills_feedback}

    **Overall Structure:**
    {structure_feedback}
    """


def generate_experience_feedback(missing_experience: List[str]) -> str:
    if missing_experience:
        experience_feedback = f"Experience: The resume mentions some key technologies, but make sure to emphasize experience with the following: "
        experience_feedback += ', '.join(missing_experience) + ". Consider including relevant projects or achievements to show your expertise."
    else:
        experience_feedback = "Experience: The resume aligns well with the experience required for the job. Keep up the good work!"
    return experience_feedback

def generate_skills_feedback(missing_skills: List[str]) -> str:
    if missing_skills:
        skills_feedback = f"Skills: Consider adding specific experience with the following skills to match the job description: {', '.join(missing_skills)}."
    else:
        skills_feedback = "Skills: The resume lists many of the relevant skills required for the job."
    return skills_feedback

def generate_structure_feedback(resume_text: str) -> str:
    if "certifications" not in resume_text.lower():
        structure_feedback = "Overall Structure: The resume could benefit from a section highlighting relevant certifications or training courses."
    elif "education" not in resume_text.lower():
        structure_feedback = "Overall Structure: Ensure that the education section is clearly outlined if it is relevant to the job."
    else:
        structure_feedback = "Overall Structure: The resume is well-structured, but ensuring that relevant certifications or training courses are emphasized can further enhance its impact."
    return structure_feedback

def clean_feedback(raw_feedback: str) -> Dict:
    """
    Parse raw feedback to extract specific feedback sections.
    """
    return {match.group(1): match.group(2).strip() for match in SECTION_REGEX.finditer(raw_feedback)}

def refine_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Refine extracted entities by categorizing technologies and tools
    into the correct labels.
    """
    refined_entities = []
    technology_keywords = {'Docker', 'Kubernetes', 'AWS', 'GCP', 'Flask', 'Django', 'FastAPI'}  # Add any other technologies
    for ent in entities:
        # Correct misclassified entities
        if ent["label"] == "PERSON" and ent["text"] in technology_keywords:
            ent["label"] = "TECHNOLOGY"  # Correct misclassification
        elif ent["label"] == "ORG" and ent["text"] in technology_keywords:
            ent["label"] = "TOOL"  # Reclassify as tool if it's a known tool
        elif ent["label"] == "GPE" and ent["text"] in technology_keywords:
            ent["label"] = "TECHNOLOGY"  # Fix misclassification as geopolitical entity

        refined_entities.append(ent)

    return refined_entities

def extract_named_entities(resume_text: str) -> List[Dict[str, str]]:
    """
    Extract and refine named entities using SpaCy and customized corrections.
    """
    spacy_model = cached_spacy_model()
    doc = spacy_model(resume_text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return refine_entities(entities)


def analyze_keywords_with_spacy(resume_text: str) -> Dict:
    """
    Extract keywords and entities using SpaCy.
    """
    spacy_model = cached_spacy_model()
    doc = spacy_model(resume_text)

    keywords = [
        token.lemma_ for token in doc
        if token.is_alpha and token.pos_ in {"NOUN", "PROPN"} and token.lemma_.lower() not in STOP_WORDS
    ]
    entities = extract_named_entities(resume_text)

    return {
        "keywords": Counter(keywords).most_common(10),
        "entities": entities,
    }

def validate_feedback(response: Dict) -> Dict:
    """
    Validate feedback to ensure all sections are included with default text if missing.
    """
    expected_sections = ["ATS Compatibility", "Experience", "Skills", "Overall Structure"]
    validated_feedback = {section: response.get(section, "No specific feedback provided.") for section in expected_sections}

    return validated_feedback

def analyze_resume(resume_text: str, job_description: str) -> Dict:
    """
    Analyze the resume and generate feedback, keyword analysis, and ATS score.
    """
    try:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Parallelize SpaCy and DistilBERT tasks
            spacy_future = executor.submit(analyze_keywords_with_spacy, resume_text)
            distilbert_future = executor.submit(generate_feedback, resume_text, job_description)

            # Get results from both futures
            spacy_results = spacy_future.result()
            feedback = distilbert_future.result()

        if not feedback:
            raise ValueError("DistilBERT returned empty feedback.")

        cleaned_feedback = clean_feedback(feedback)
        validated_feedback = validate_feedback(cleaned_feedback)

        ats_score = calculate_ats_score(
            [word for word, _ in spacy_results["keywords"]],
            [word for word, _ in analyze_keywords_with_spacy(job_description)["keywords"]]
        )

        return {
            "word_count": len(resume_text.split()),
            "sentence_count": len(re.findall(r'\.', resume_text)),
            "top_keywords": [{"word": word, "count": count} for word, count in spacy_results["keywords"]],
            "named_entities": spacy_results["entities"],
            "ats_score": ats_score,
            "feedback": validated_feedback,
        }
    except Exception as e:
        logging.error(f"Error during resume analysis: {str(e)}")
        return {"error": str(e)}


