"""
Data Preprocessing Module for Resume Parser
Uses NLTK for text cleaning and tokenization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ResumePreprocessor:
    """Preprocesses resume text data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # Add domain-specific stop words
        self.stop_words.update(['resume', 'cv', 'curriculum', 'vitae', 'experience', 
                               'work', 'job', 'position', 'company', 'years', 'year',
                               'month', 'months', 'detail', 'details', 'skill', 'skills'])
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_skills_keywords(self, text):
        """Extract potential skills and technical keywords"""
        # Common technical terms and skills patterns
        skill_patterns = [
            r'\b(python|java|javascript|sql|html|css|react|angular|node|django|flask)\b',
            r'\b(machine learning|deep learning|nlp|ai|artificial intelligence)\b',
            r'\b(data science|data analysis|data visualization)\b',
            r'\b(aws|azure|gcp|cloud computing)\b',
            r'\b(docker|kubernetes|git|jenkins|ci/cd)\b',
            r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch)\b',
            r'\b(agile|scrum|devops|automation)\b',
            r'\b(mysql|mongodb|postgresql|redis)\b',
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize text and lemmatize words"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return lemmatized
    
    def preprocess_resume(self, resume_text):
        """Complete preprocessing pipeline for a resume"""
        # Clean text
        cleaned = self.clean_text(resume_text)
        
        # Extract skills
        skills = self.extract_skills_keywords(cleaned)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        # Combine tokens and skills
        processed_text = ' '.join(tokens + skills)
        
        return {
            'cleaned_text': cleaned,
            'tokens': tokens,
            'skills': skills,
            'processed_text': processed_text
        }
    
    def preprocess_dataset(self, df):
        """Preprocess entire dataset"""
        processed_data = []
        
        for idx, row in df.iterrows():
            resume_text = row['Resume']
            category = row['Category']
            
            processed = self.preprocess_resume(resume_text)
            processed['category'] = category
            processed['original_text'] = resume_text
            
            processed_data.append(processed)
        
        return processed_data
