import logging
from typing import Dict, List, Union
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from functools import lru_cache

from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

from src.spacy_model import get_spacy_model  # Assuming this loads a SpaCy model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compile regex patterns for performance
SECTION_REGEX = re.compile(r"(ATS Compatibility|Experience|Skills|Overall Structure):\s*(.*?)\s*(?=\*\*|$)", re.DOTALL)


@lru_cache(maxsize=None)
def cached_spacy_model():
    """
    Cache the SpaCy model for faster reuse.
    """
    return get_spacy_model()


@lru_cache(maxsize=None)
def cached_distilbert_model():
    """
    Cache the DistilBERT model and tokenizer for faster reuse.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


def get_distilbert_embeddings(text: str) -> torch.Tensor:
    """
    Generate embeddings for a given text using DistilBERT.
    """
    try:
        tokenizer, model = cached_distilbert_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise


def calculate_ats_score(resume_keywords: List[str], job_desc_keywords: List[str]) -> float:
    """
    Calculate ATS score based on common keywords between the resume and job description.
    """
    common_keywords = set(resume_keywords) & set(job_desc_keywords)
    total_keywords = set(job_desc_keywords)
    return (len(common_keywords) / len(total_keywords)) * 100 if total_keywords else 0.0


def extract_named_entities(resume_text: str) -> List[Dict[str, str]]:
    """
    Extract and refine named entities using SpaCy.
    """
    spacy_model = cached_spacy_model()
    doc = spacy_model(resume_text)

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    refined_entities = [
        {
            "text": ent["text"],
            "label": "TECHNOLOGY" if ent["text"].isalpha() else ent["label"],
        }
        for ent in entities
    ]
    return refined_entities


def analyze_keywords_with_spacy(text: str) -> Dict[str, Union[List, Counter]]:
    """
    Analyze keywords and named entities using SpaCy.
    """
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


def generate_feedback(resume_text: str, job_description: str) -> Dict[str, str]:
    """
    Generate feedback by comparing resume and job description content.
    """
    try:
        # Get embeddings and calculate similarity
        resume_embedding = get_distilbert_embeddings(resume_text)
        job_desc_embedding = get_distilbert_embeddings(job_description)
        similarity_score = cosine_similarity(resume_embedding, job_desc_embedding)[0][0]

        ats_feedback = f"The resume's alignment with the job description is {similarity_score * 100:.2f}%."

        # Analyze keywords
        resume_analysis = analyze_keywords_with_spacy(resume_text)
        job_desc_analysis = analyze_keywords_with_spacy(job_description)

        # Identify missing skills/experience
        resume_keywords = [word for word, _ in resume_analysis["keywords"]]
        job_desc_keywords = [word for word, _ in job_desc_analysis["keywords"]]
        missing_keywords = list(set(job_desc_keywords) - set(resume_keywords))

        skills_feedback = (
            f"Consider emphasizing these missing skills/experiences: {', '.join(missing_keywords)}."
            if missing_keywords else "The resume matches most of the required skills and experiences."
        )

        # Structural Feedback
        structure_feedback = (
            "Ensure your resume has sections on certifications, training, and education."
            if "certifications" not in resume_text.lower() or "education" not in resume_text.lower()
            else "The resume structure is clear and complete."
        )

        return {
            "ATS Compatibility": ats_feedback,
            "Experience and Skills": skills_feedback,
            "Overall Structure": structure_feedback,
        }

    except Exception as e:
        logging.error(f"Error generating feedback: {str(e)}")
        raise


def analyze_resume(resume_text: str, job_description: str) -> Dict:
    """
    Analyze the resume and generate structured feedback.
    """
    try:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = {
                "spacy": executor.submit(analyze_keywords_with_spacy, resume_text),
                "feedback": executor.submit(generate_feedback, resume_text, job_description),
            }

            done, _ = wait(futures.values(), return_when=FIRST_EXCEPTION)

            for future in done:
                if future.exception():
                    raise future.exception()

            spacy_results = futures["spacy"].result()
            feedback = futures["feedback"].result()

        # Calculate ATS score
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
            "feedback": feedback,
        }

    except Exception as e:
        logging.error(f"Error during resume analysis: {str(e)}")
        return {"error": str(e)}
