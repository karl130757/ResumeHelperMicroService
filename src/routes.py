import os
import magic  # Import the magic library for MIME type detection
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

        # Read the file into memory to detect its type
        file_content = resume_file.read()
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file_content)

        
        
        # Map MIME type to an extension (you can expand this mapping as needed)
        extension = {
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
        }.get(mime_type, "")  # Default to no extension if not recognized

        if not extension:
            return jsonify({"error": f"Unsupported file type: {mime_type}"}), 400

        # Save the file with the correct extension
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], resume_file.filename + extension)
        with open(file_path, "wb") as f:
            f.write(file_content)

        resume_text = parse_file(file_path)

       
        os.remove(file_path)  # Cleanup
    else:
        resume_text = request.form.get("resume_text")

    job_description = request.form.get("job_description")

   
    if not resume_text or not job_description:
        return jsonify({"error": "Both resume_text and job_description are required"}), 400

    result = analyze_resume(resume_text, job_description)
    return jsonify(result)
