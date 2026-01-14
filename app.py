"""
Flask Web Application for Resume Parser AI
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from resume_parser import ResumeParserAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'resume-parser-secret-key'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Initialize parser
parser = ResumeParserAI()

# Load model if it exists
model_path = 'resume_parser_model.pkl'
if os.path.exists(model_path):
    try:
        parser.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'python main.py' first to train the model.")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_text_file(filepath):
    """Read text from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return None


def read_pdf_file(filepath):
    """Read text from PDF file"""
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return None


def read_docx_file(filepath):
    """Read text from DOCX file"""
    try:
        from docx import Document
        doc = Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return None


def extract_text_from_file(filepath, filename):
    """Extract text from various file formats"""
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext == 'txt':
        return read_text_file(filepath)
    elif ext == 'pdf':
        return read_pdf_file(filepath)
    elif ext in ['doc', 'docx']:
        return read_docx_file(filepath)
    else:
        return None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and score resume"""
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['resume']
        category = request.form.get('category', 'General')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .txt, .pdf, .doc, or .docx files'}), 400
        
        # Save uploaded file
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from file
        resume_text = extract_text_from_file(filepath, filename)
        
        if not resume_text or len(resume_text.strip()) < 50:
            os.remove(filepath)  # Clean up
            return jsonify({'error': 'Could not extract text from file or file is too short'}), 400
        
        # Check if model is trained
        if not parser.is_trained:
            os.remove(filepath)  # Clean up
            return jsonify({
                'error': 'Model not trained. Please run "python main.py" first to train the model.',
                'needs_training': True
            }), 400
        
        # Score the resume
        result = parser.predict_single_resume(resume_text, category)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'score': round(result['score'], 2),
            'skill_score': round(result['skill_score'], 2),
            'quality_score': round(result['quality_score'], 2),
            'relevance_score': round(result['relevance_score'], 2),
            'cluster': int(result['cluster']),
            'skills': result['processed_data']['skills'][:15],  # Top 15 skills
            'category': category,
            'feedback': result.get('feedback', {})
        }
        
        return jsonify(response)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)  # Clean up on error
        return jsonify({'error': f'Error processing resume: {str(e)}'}), 500


@app.route('/text', methods=['POST'])
def process_text():
    """Process resume text directly (paste text)"""
    try:
        data = request.get_json()
        resume_text = data.get('text', '')
        category = data.get('category', 'General')
        
        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Resume text is too short (minimum 50 characters)'}), 400
        
        # Check if model is trained
        if not parser.is_trained:
            return jsonify({
                'error': 'Model not trained. Please run "python main.py" first to train the model.',
                'needs_training': True
            }), 400
        
        # Score the resume
        result = parser.predict_single_resume(resume_text, category)
        
        # Prepare response
        response = {
            'success': True,
            'score': round(result['score'], 2),
            'skill_score': round(result['skill_score'], 2),
            'quality_score': round(result['quality_score'], 2),
            'relevance_score': round(result['relevance_score'], 2),
            'cluster': int(result['cluster']),
            'skills': result['processed_data']['skills'][:15],  # Top 15 skills
            'category': category,
            'feedback': result.get('feedback', {})
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Error processing resume: {str(e)}'}), 500


@app.route('/categories', methods=['GET'])
def get_categories():
    """Get list of available job categories"""
    try:
        if os.path.exists('UpdatedResumeDataSet.csv'):
            df = pd.read_csv('UpdatedResumeDataSet.csv')
            categories = sorted(df['Category'].unique().tolist())
            return jsonify({'categories': categories})
        else:
            return jsonify({'categories': [
                'Data Science', 'Java Developer', 'Web Designing', 'HR',
                'Business Analyst', 'SAP Developer', 'Automation Testing',
                'Mechanical Engineer', 'Civil Engineer', 'Electrical Engineering',
                'General'
            ]})
    except Exception as e:
        return jsonify({'categories': ['General']})


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    print("\n" + "=" * 60)
    print("Resume Parser AI - Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
