from src.spacy_model import get_spacy_model
from src.gpt_model import get_gpt_model


def generate_feedback(resume_text, job_description):
    prompt = f"""
    You are an AI-powered resume analyzer. Given the following resume and job description, provide actionable feedback to help the resume pass an ATS scan and align with the job description. Focus on specific improvements related to keywords, skills, experience, and structure.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Please provide concise, actionable feedback in these sections:

    1. **ATS Compatibility**:
       - Suggest specific keywords and phrases the resume should include to improve ATS compatibility.
       - Identify industry-specific terms or key qualifications missing from the resume.

    2. **Experience**:
       - Suggest how to improve or highlight relevant work experience.
       - Recommend adding quantifiable achievements or action verbs to make the experience more impactful.

    3. **Skills**:
       - Recommend additional technical and soft skills to include based on the job description.
       - Suggest any certifications or tools that should be added.

    4. **Overall Structure**:
       - Provide suggestions to improve the overall structure of the resume (e.g., section order, clarity, layout).
       - Recommend adding sections like "Projects" or "Achievements" if relevant.

    Focus on giving direct, actionable advice that will improve the resume's ATS performance and alignment with the job description.
    """
    return prompt









def analyze_resume(resume_text, job_description):
    # ATS scoring
    spacy_model = get_spacy_model()
    doc = spacy_model(resume_text)
    ats_score = len([ent for ent in doc.ents])  # Example scoring logic

    # Feedback generation with GPT-J
    gpt_model = get_gpt_model()
    prompt = generate_feedback(resume_text,job_description)
    # prompt = (
    #     f"Resume:\n{resume_text}\n\n"
    #     f"Job Description:\n{job_description}\n\n"
    #     "Provide one actionable feedback to improve the resume."
    # )
    feedback = gpt_model(prompt, max_length=200, num_return_sequences=1)

    return {
        "ats_score": ats_score,
        "feedback": feedback[0]["generated_text"]
    }
