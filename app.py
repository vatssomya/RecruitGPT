import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from resume_processor import ResumeProcessor
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_change_in_production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize resume processor
resume_processor = ResumeProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and process resumes"""
    try:
        # Check if job description is provided
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            flash('Please provide a job description.', 'error')
            return redirect(url_for('index'))

        # Check if files are uploaded
        if 'resume_files' not in request.files:
            flash('No resume files selected.', 'error')
            return redirect(url_for('index'))

        files = request.files.getlist('resume_files')
        if not files or all(file.filename == '' for file in files):
            flash('No resume files selected.', 'error')
            return redirect(url_for('index'))

        # Process uploaded files
        uploaded_files = []
        for file in files:
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Add timestamp to avoid conflicts
                import time
                timestamp = str(int(time.time()))
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append((filepath, file.filename))
            elif file and file.filename != '':
                flash(f'File type not supported: {file.filename}', 'warning')

        if not uploaded_files:
            flash('No valid resume files uploaded. Supported formats: PDF, DOCX, TXT', 'error')
            return redirect(url_for('index'))

        # Process resumes and calculate rankings
        app.logger.info(f"Processing {len(uploaded_files)} resumes against job description")
        results = resume_processor.rank_resumes(job_description, uploaded_files)

        # Clean up uploaded files
        for filepath, _ in uploaded_files:
            try:
                os.remove(filepath)
            except OSError:
                pass

        if not results:
            flash('No resumes could be processed successfully. Please check file formats and content.', 'error')
            return redirect(url_for('index'))

        return render_template('results.html', 
                             results=results, 
                             job_description=job_description,
                             total_resumes=len(uploaded_files))

    except Exception as e:
        app.logger.error(f"Error processing uploads: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash(f'An error occurred while processing files: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File size too large. Maximum size is 16MB per file.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    app.logger.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
