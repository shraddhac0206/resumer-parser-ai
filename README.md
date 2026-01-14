# Resume Parser AI

An intelligent resume parsing system that uses Natural Language Processing (NLP) and Machine Learning techniques to analyze resumes and assign scores from 0-10 based on skills, content quality, and relevance.

## Project Overview

This project automates the resume screening process by:
- **Preprocessing** resume text using NLTK
- **Extracting features** using TF-IDF and keyword analysis
- **Clustering** resumes to group similar candidates
- **Scoring** resumes from 0-10 based on multiple factors

## Features

- ✅ Text preprocessing with NLTK (tokenization, lemmatization, stopword removal)
- ✅ Feature extraction using TF-IDF vectorization
- ✅ Clustering algorithm to group related skills and resumes
- ✅ Intelligent scoring system (0-10) based on:
  - Technical skills and keywords
  - Content quality and diversity
  - Category relevance
- ✅ Support for 25+ job categories
- ✅ Model persistence (save/load trained models)
- ✅ **Web Application** - Beautiful browser interface to upload and score resumes

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (will be done automatically on first run):
   - punkt tokenizer
   - stopwords
   - wordnet
   - averaged_perceptron_tagger

## Dataset

The project uses the **UpdatedResumeDataSet.csv** file which contains:
- **Category**: Job title/category (25 unique categories)
- **Resume**: Full resume text

The dataset includes resumes for categories like:
- Data Science
- Java Developer
- Web Designing
- HR
- Business Analyst
- And 20+ more categories

## Usage

### 🌐 Web Application (Recommended)

**Start the web application for an easy-to-use browser interface:**

1. **First, train the model** (one-time setup):
   ```bash
   python main.py
   ```

2. **Start the web server**:
   ```bash
   python app.py
   ```

3. **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

4. **Upload your resume** or paste the text:
   - Choose your job category
   - Upload a file (.txt, .pdf, .doc, .docx) or paste resume text
   - Click "Analyze Resume"
   - View your score and detailed breakdown!

**Features:**
- 📤 Upload resume files (TXT, PDF, DOC, DOCX)
- 📝 Paste resume text directly
- 📊 Beautiful score visualization
- 🔍 Detected skills display
- 📈 Detailed score breakdown (Skill, Quality, Relevance)

### Command Line Usage

Run the main script to train the model and analyze all resumes:

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Extract features from resumes
3. Perform clustering
4. Score all resumes
5. Display top 10 resumes
6. Save results to `resume_scores.csv`
7. Save the trained model to `resume_parser_model.pkl`
8. Generate visualization charts

### Using the Trained Model

```python
from resume_parser import ResumeParserAI

# Load the trained model
parser = ResumeParserAI()
parser.load_model('resume_parser_model.pkl')

# Score a new resume
resume_text = """
Your resume text here...
"""

result = parser.predict_single_resume(resume_text, category='Data Science')

print(f"Score: {result['score']:.2f}/10")
print(f"Skill Score: {result['skill_score']:.2f}")
print(f"Quality Score: {result['quality_score']:.2f}")
print(f"Relevance Score: {result['relevance_score']:.2f}")
print(f"Cluster: {result['cluster']}")
```

### Custom Usage

```python
from resume_parser import ResumeParserAI
import pandas as pd

# Initialize
parser = ResumeParserAI()

# Load your data
df = pd.read_csv('your_resume_data.csv')

# Train
results = parser.train('UpdatedResumeDataSet.csv', n_clusters=10)

# Get top resumes
top_resumes = parser.get_top_resumes(results['df'], n=20)

# Save model for later use
parser.save_model('my_model.pkl')
```

## Project Structure

```
Resume Parser/
│
├── UpdatedResumeDataSet.csv      # Dataset file
├── main.py                        # Main execution script
├── app.py                         # Flask web application
├── resume_parser.py               # Main parser class
├── preprocessing.py               # Text preprocessing module
├── feature_extraction.py          # Feature extraction module
├── clustering_scoring.py          # Clustering and scoring module
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── templates/
│   └── index.html                # Web interface HTML
├── static/
│   └── style.css                 # Web interface styling
│
├── resume_parser_model.pkl        # Saved model (after training)
├── resume_scores.csv              # Results (after training)
└── resume_analysis.png            # Visualization (after training)
```

## Scoring System

The scoring system (0-10) considers:

1. **Skill Score (0-4 points)**
   - Programming languages (Python, Java, JavaScript, etc.)
   - Web development skills (React, Angular, Django, etc.)
   - Data science tools (Pandas, NumPy, Scikit-learn, etc.)
   - Databases (MySQL, MongoDB, PostgreSQL, etc.)
   - Cloud technologies (AWS, Azure, Docker, etc.)
   - ML/AI skills (Machine Learning, Deep Learning, NLP, etc.)
   - Development tools (Git, Agile, DevOps, etc.)

2. **Quality Score (0-3 points)**
   - Resume length (optimal: 500-5000 characters)
   - Token diversity (unique words / total words)
   - Number of skills mentioned

3. **Relevance Score (0-3 points)**
   - Category-specific keyword matching
   - Domain relevance

## Clustering

The system uses K-Means clustering to group resumes with similar:
- Skills and technologies
- Content patterns
- Domain expertise

This helps identify:
- Candidates with similar skill sets
- Resume patterns by category
- Skill clusters in the dataset

## Output Files

After running `main.py`, you'll get:

1. **resume_scores.csv**: All resumes with their scores and cluster assignments
2. **resume_parser_model.pkl**: Trained model for future predictions
3. **resume_analysis.png**: Visualization charts showing score distribution

## Requirements

- Python 3.7+
- pandas
- nltk
- scikit-learn
- numpy
- matplotlib
- seaborn
- flask (for web application)
- PyPDF2 (for PDF file support)
- python-docx (for DOCX file support)

## How It Works

1. **Preprocessing**: 
   - Cleans text (removes URLs, emails, special characters)
   - Tokenizes and lemmatizes words
   - Removes stopwords
   - Extracts technical skills

2. **Feature Extraction**:
   - TF-IDF vectorization (captures important terms)
   - Keyword-based features (skill categories)
   - Text statistics (length, token count)

3. **Clustering**:
   - Groups similar resumes using K-Means
   - Identifies skill patterns

4. **Scoring**:
   - Calculates weighted scores based on skills, quality, and relevance
   - Normalizes to 0-10 scale

## Future Enhancements

- [x] Support for PDF resume parsing
- [x] Web interface
- [ ] Experience level detection
- [ ] Education extraction
- [ ] Multi-language support
- [ ] API endpoint

## License

This project is for educational purposes.

## Author

Built as a beginner-friendly AI/ML project for learning Natural Language Processing and Machine Learning techniques.

---

**Note**: This is a learning project. For production use, consider additional validation, error handling, and model tuning.
