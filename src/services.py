import logging
from typing import Dict, List
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from src.spacy_model import get_spacy_model
from src.gpt_model import get_gpt_model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from spacy.lang.en.stop_words import STOP_WORDS

# Compile regex patterns for performance
SECTION_REGEX = re.compile(r"(ATS Compatibility|Experience|Skills|Overall Structure):\s*(.*?)\s*(?=\*\*|$)", re.DOTALL)

@lru_cache(maxsize=None)
def cached_spacy_model():
    """Cache the SpaCy model for faster reuse."""
    return get_spacy_model()

@lru_cache(maxsize=None)
def cached_gpt_model():
    """Cache the GPT model for faster reuse."""
    return get_gpt_model()

def generate_feedback(resume_text: str, job_description: str) -> str:
    """
    Generate a GPT prompt for feedback on the resume against the job description.
    """
    prompt = f"""
    You are an AI-powered resume analyzer. 
    Your goal is to evaluate the following resume and job description and provide actionable feedback to improve the resume's ATS compatibility and relevance to the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Provide detailed feedback in the following sections. If you cannot infer specific details for a section, provide general improvement suggestions relevant to the topic.

    1. **ATS Compatibility**:
       - List specific keywords and phrases the resume should include for better ATS performance.
    2. **Experience**:
       - Suggest improvements to highlight relevant work experience.
    3. **Skills**:
       - Recommend additional technical and soft skills based on the job description.
    4. **Overall Structure**:
       - Provide recommendations to improve the section order, layout, or clarity.

    Ensure the response is formatted as JSON with keys for each section:
    {{
        "ATS Compatibility": "<specific feedback>",
        "Experience": "<specific feedback>",
        "Skills": "<specific feedback>",
        "Overall Structure": "<specific feedback>"
    }}
    """
    return prompt

def clean_feedback(raw_feedback: str) -> Dict:
    """
    Parse raw feedback to extract specific feedback sections.
    """
    return {match.group(1): match.group(2).strip() for match in SECTION_REGEX.finditer(raw_feedback)}

def refine_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter and refine extracted entities.
    """
    return [
        ent for ent in entities
        if not (ent["label"] == "ORG" and ("Key Responsibilities" in ent["text"] or ent["text"].startswith("�")))
        and not (ent["label"] == "MONEY" and re.match(r"^\d{3,}$", ent["text"]))
    ]

def extract_named_entities(resume_text: str) -> List[Dict[str, str]]:
    """
    Extract and refine named entities using SpaCy.
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
    Validate GPT feedback, ensuring all sections are included with default text if missing.
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
            # Parallelize SpaCy and GPT tasks
            spacy_future = executor.submit(analyze_keywords_with_spacy, resume_text)
            gpt_future = executor.submit(
                cached_gpt_model(),
                generate_feedback(resume_text, job_description),
                max_length=500,
                num_return_sequences=1,
            )

            # Get results from both futures
            spacy_results = spacy_future.result()
            feedback = gpt_future.result()

        if not feedback:
            raise ValueError("GPT model returned empty feedback.")

        cleaned_feedback = clean_feedback(feedback[0]["generated_text"])
        validated_feedback = validate_feedback(cleaned_feedback)

        ats_score = sum(freq for _, freq in spacy_results["keywords"]) * 10

        return {
            "word_count": len(resume_text.split()),
            "sentence_count": len(re.findall(r'\.', resume_text)),
            "top_keywords": [{"word": word, "count": count} for word, count in spacy_results["keywords"]],
            "named_entities": spacy_results["entities"],
            "ats_score": ats_score,
            "feedback": validated_feedback,
        }
    except Exception as e:
        logging.error(f"Error analyzing resume: {e}")
        return {"error": str(e)}
