"""
Feature Extraction Module
Uses TF-IDF and word embeddings for feature extraction
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re


class FeatureExtractor:
    """Extract features from preprocessed resumes"""
    
    def __init__(self, max_features=5000, n_components=100):
        self.max_features = max_features
        self.n_components = n_components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        self.scaler_fitted = False
        self.pca_fitted = False
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features from text"""
        if not self.is_fitted:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix.toarray()
    
    def extract_keyword_features(self, processed_data):
        """Extract keyword-based features"""
        # Common skill categories
        skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'express'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins'],
            'ml_ai': ['machine learning', 'deep learning', 'neural network', 'nlp', 'ai', 'computer vision'],
            'tools': ['git', 'jira', 'agile', 'scrum', 'devops', 'ci/cd']
        }
        
        features = []
        for data in processed_data:
            text_lower = data['processed_text'].lower()
            feature_vector = []
            
            for category, keywords in skill_categories.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
            
            # Add skill count
            feature_vector.append(len(data['skills']))
            
            # Add text length features
            feature_vector.append(len(data['tokens']))
            feature_vector.append(len(data['cleaned_text']))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_all_features(self, processed_data, fit_transform=True):
        """Extract all features (TF-IDF + keyword features)
        
        Args:
            processed_data: List of processed resume data
            fit_transform: If True, fit scaler and PCA (for training).
                          If False, use already-fitted models (for prediction)
        """
        texts = [data['processed_text'] for data in processed_data]
        
        # TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Keyword features
        keyword_features = self.extract_keyword_features(processed_data)
        
        # Combine features
        all_features = np.hstack([tfidf_features, keyword_features])
        
        # Normalize
        # Check if scaler is actually fitted (backward compatibility)
        scaler_is_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
        
        if fit_transform or not (self.scaler_fitted or scaler_is_fitted):
            all_features = self.scaler.fit_transform(all_features)
            self.scaler_fitted = True
        else:
            all_features = self.scaler.transform(all_features)
            if not self.scaler_fitted:
                self.scaler_fitted = True  # Mark as fitted after successful transform
        
        # Apply PCA for dimensionality reduction
        n_samples = all_features.shape[0]
        n_features = all_features.shape[1]
        
        # Check if PCA is actually fitted (backward compatibility)
        pca_is_fitted = hasattr(self.pca, 'components_') and self.pca.components_ is not None
        
        if fit_transform and n_samples > 1:
            # During training with multiple samples
            if not (self.pca_fitted or pca_is_fitted):
                # Adjust n_components to be valid
                max_components = min(self.n_components, n_samples - 1, n_features)
                if max_components < self.n_components:
                    self.pca = PCA(n_components=max_components)
                all_features = self.pca.fit_transform(all_features)
                self.pca_fitted = True
            else:
                # Already fitted, just transform
                all_features = self.pca.transform(all_features)
        elif not fit_transform:
            # During prediction - use already-fitted PCA
            if self.pca_fitted or pca_is_fitted:
                all_features = self.pca.transform(all_features)
                if not self.pca_fitted:
                    self.pca_fitted = True  # Mark as fitted after successful transform
            else:
                # PCA not fitted yet - this shouldn't happen if model is trained
                # But handle gracefully by returning features as-is
                pass
        else:
            # Single sample during training - skip PCA
            # This shouldn't happen in normal flow, but handle it gracefully
            pass
        
        return all_features
    
    def get_feature_names(self):
        """Get names of features"""
        tfidf_features = [f'tfidf_{i}' for i in range(self.max_features)]
        keyword_features = [
            'programming_skills', 'web_dev_skills', 'data_science_skills',
            'database_skills', 'cloud_skills', 'ml_ai_skills', 'tools_skills',
            'total_skills', 'token_count', 'text_length'
        ]
        return tfidf_features + keyword_features
