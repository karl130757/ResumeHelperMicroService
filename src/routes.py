import os
from flask import Blueprint, request, jsonify, current_app
from src.parsers import parse_file
from src.services import analyze_resume

analysis_bp = Blueprint("analysis", __name__)

@analysis_bp.route("/analyze", methods=["POST"])
def analyze():
    if "resume_file" in request.files:
        resume_file = request.files["resume_file"]
        if not resume_file.filename:
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(file_path)
        resume_text = parse_file(file_path)
        os.remove(file_path)  # Cleanup
    else:
        resume_text = request.form.get("resume_text")

    job_description = request.form.get("job_description")
    if not resume_text or not job_description:
        return jsonify({"error": "Both resume_text and job_description are required"}), 400

    result = analyze_resume(resume_text, job_description)
    return jsonify(result)
