"""
Main Resume Parser AI System
Combines preprocessing, feature extraction, clustering, and scoring
"""

import pandas as pd
import numpy as np
import pickle
import os
from preprocessing import ResumePreprocessor
from feature_extraction import FeatureExtractor
from clustering_scoring import ResumeClustering, ResumeScorer


class ResumeParserAI:
    """Main Resume Parser AI System"""
    
    def __init__(self):
        self.preprocessor = ResumePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.clustering = ResumeClustering()
        self.scorer = ResumeScorer()
        self.is_trained = False
        self.processed_data = None
        self.features = None
        self.cluster_labels = None
        self.scores = None
    
    def load_data(self, csv_path):
        """Load resume dataset from CSV"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} resumes with {df['Category'].nunique()} categories")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        print("Preprocessing resumes...")
        self.processed_data = self.preprocessor.preprocess_dataset(df)
        print(f"Preprocessed {len(self.processed_data)} resumes")
        return self.processed_data
    
    def extract_features(self, processed_data):
        """Extract features from processed data"""
        print("Extracting features...")
        # Use fit_transform=True for training (multiple samples)
        self.features = self.feature_extractor.extract_all_features(processed_data, fit_transform=True)
        print(f"Extracted features with shape: {self.features.shape}")
        return self.features
    
    def perform_clustering(self, features, n_clusters=None):
        """Perform clustering on resumes"""
        print("Performing clustering...")
        if n_clusters is None:
            n_clusters = min(10, len(features) // 10)
        
        self.clustering.fit(features, n_clusters=n_clusters)
        self.cluster_labels = self.clustering.predict(features)
        print(f"Created {n_clusters} clusters")
        
        # Get cluster keywords
        cluster_keywords = self.clustering.get_cluster_keywords(
            self.processed_data, self.cluster_labels
        )
        
        print("\nCluster Keywords:")
        for cluster_id, keywords in cluster_keywords.items():
            print(f"Cluster {cluster_id}: {', '.join(keywords[:5])}")
        
        return self.cluster_labels
    
    def score_resumes(self, processed_data, categories):
        """Score all resumes"""
        print("Scoring resumes...")
        self.scores = self.scorer.score_resumes(processed_data, categories)
        print("Scoring completed")
        return self.scores
    
    def train(self, csv_path, n_clusters=None):
        """Complete training pipeline"""
        # Load data
        df = self.load_data(csv_path)
        
        # Preprocess
        processed_data = self.preprocess_data(df)
        
        # Extract features
        features = self.extract_features(processed_data)
        
        # Clustering
        cluster_labels = self.perform_clustering(features, n_clusters)
        
        # Scoring
        categories = df['Category'].tolist()
        scores = self.score_resumes(processed_data, categories)
        
        self.is_trained = True
        
        return {
            'df': df,
            'processed_data': processed_data,
            'features': features,
            'cluster_labels': cluster_labels,
            'scores': scores
        }
    
    def predict_single_resume(self, resume_text, category='General'):
        """Predict score for a single resume"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess
        processed = self.preprocessor.preprocess_resume(resume_text)
        
        # Extract features (use transform only, not fit_transform)
        features = self.feature_extractor.extract_all_features([processed], fit_transform=False)
        
        # Predict cluster
        cluster = self.clustering.predict(features)[0]
        
        # Score
        scores = self.scorer.score_resumes([processed], [category])
        
        return {
            'score': scores['final_scores'][0],
            'cluster': cluster,
            'skill_score': scores['skill_scores'][0],
            'quality_score': scores['quality_scores'][0],
            'relevance_score': scores['relevance_scores'][0],
            'processed_data': processed,
            'feedback': scores['feedbacks'][0] if 'feedbacks' in scores else {}
        }
    
    def get_top_resumes(self, df, n=10):
        """Get top N resumes by score"""
        if self.scores is None:
            raise ValueError("Model must be trained and scored first")
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Score'] = self.scores['final_scores']
        results_df['Skill_Score'] = self.scores['skill_scores']
        results_df['Quality_Score'] = self.scores['quality_scores']
        results_df['Relevance_Score'] = self.scores['relevance_scores']
        results_df['Cluster'] = self.cluster_labels
        
        # Sort by score
        top_resumes = results_df.nlargest(n, 'Score')
        
        return top_resumes
    
    def save_model(self, filepath='resume_parser_model.pkl'):
        """Save trained model"""
        model_data = {
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor,
            'clustering': self.clustering,
            'scorer': self.scorer,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='resume_parser_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.preprocessor = model_data['preprocessor']
        self.feature_extractor = model_data['feature_extractor']
        self.clustering = model_data['clustering']
        self.scorer = model_data['scorer']
        self.is_trained = model_data['is_trained']
        
        # Backward compatibility: Set fitted flags if they don't exist
        if not hasattr(self.feature_extractor, 'scaler_fitted'):
            # Check if scaler is actually fitted by looking for mean_ attribute
            self.feature_extractor.scaler_fitted = hasattr(self.feature_extractor.scaler, 'mean_')
        
        if not hasattr(self.feature_extractor, 'pca_fitted'):
            # Check if PCA is actually fitted by looking for components_ attribute
            self.feature_extractor.pca_fitted = hasattr(self.feature_extractor.pca, 'components_')
        
        print(f"Model loaded from {filepath}")
