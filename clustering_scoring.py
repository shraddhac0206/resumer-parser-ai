"""
Clustering and Scoring Module
Groups related skills and assigns scores to resumes
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re


class ResumeClustering:
    """Cluster resumes based on skills and content similarity"""
    
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_centers_ = None
        self.is_fitted = False
    
    def fit(self, features, n_clusters=None):
        """Fit clustering model"""
        if n_clusters is None:
            n_clusters = self.n_clusters or min(10, len(features) // 10)
        
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.is_fitted = True
    
    def predict(self, features):
        """Predict cluster assignments"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.kmeans.predict(features)
    
    def get_cluster_keywords(self, processed_data, cluster_labels, top_n=10):
        """Extract top keywords for each cluster"""
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_texts = [processed_data[i]['processed_text'] for i in cluster_indices]
            
            # Count word frequencies
            all_words = []
            for text in cluster_texts:
                words = text.split()
                all_words.extend(words)
            
            word_freq = Counter(all_words)
            top_keywords = [word for word, count in word_freq.most_common(top_n)]
            cluster_keywords[cluster_id] = top_keywords
        
        return cluster_keywords


class ResumeScorer:
    """Score resumes from 0-10 based on various factors"""
    
    def __init__(self):
        self.skill_weights = {
            'programming': 1.5,
            'web_dev': 1.3,
            'data_science': 1.4,
            'databases': 1.2,
            'cloud': 1.5,
            'ml_ai': 1.6,
            'tools': 1.1
        }
    
    def calculate_skill_score(self, processed_data):
        """Calculate score based on skills present"""
        skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'express'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins'],
            'ml_ai': ['machine learning', 'deep learning', 'neural network', 'nlp', 'ai', 'computer vision'],
            'tools': ['git', 'jira', 'agile', 'scrum', 'devops', 'ci/cd']
        }
        
        scores = []
        for data in processed_data:
            text_lower = data['processed_text'].lower()
            skill_score = 0
            
            for category, keywords in skill_categories.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    skill_score += self.skill_weights[category] * min(matches / len(keywords), 1.0)
            
            scores.append(skill_score)
        
        return np.array(scores)
    
    def calculate_content_quality_score(self, processed_data):
        """Calculate score based on content quality"""
        scores = []
        
        for data in processed_data:
            score = 0
            
            # Length score (not too short, not too long)
            text_length = len(data['cleaned_text'])
            if 500 <= text_length <= 5000:
                score += 2
            elif 200 <= text_length < 500 or 5000 < text_length <= 10000:
                score += 1
            
            # Token diversity (unique tokens / total tokens)
            tokens = data['tokens']
            if len(tokens) > 0:
                unique_ratio = len(set(tokens)) / len(tokens)
                score += unique_ratio * 2
            
            # Skills count
            skills_count = len(data['skills'])
            if skills_count >= 10:
                score += 2
            elif skills_count >= 5:
                score += 1
            
            scores.append(score)
        
        return np.array(scores)
    
    def calculate_relevance_score(self, processed_data, category):
        """Calculate relevance score based on category match"""
        # Category-specific keywords
        category_keywords = {
            'Data Science': ['python', 'machine learning', 'data science', 'pandas', 'numpy', 'scikit-learn'],
            'Java Developer': ['java', 'spring', 'hibernate', 'j2ee', 'jsp', 'servlet'],
            'Web Designing': ['html', 'css', 'javascript', 'react', 'angular', 'ui', 'ux'],
            'HR': ['recruitment', 'hiring', 'talent', 'hr', 'human resources'],
            'Business Analyst': ['business analysis', 'requirements', 'stakeholder', 'analytics'],
            'SAP Developer': ['sap', 'abap', 'sap hana', 'sap fico'],
            'Automation Testing': ['selenium', 'testing', 'automation', 'qa', 'test automation'],
            'Mechanical Engineer': ['cad', 'solidworks', 'mechanical', 'engineering', 'design'],
            'Civil Engineer': ['civil', 'construction', 'structural', 'engineering'],
            'Electrical Engineering': ['electrical', 'circuit', 'power', 'electronics']
        }
        
        scores = []
        keywords = category_keywords.get(category, [])
        
        for data in processed_data:
            text_lower = data['processed_text'].lower()
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            relevance_score = min(matches / max(len(keywords), 1) * 3, 3) if keywords else 1.5
            scores.append(relevance_score)
        
        return np.array(scores)
    
    def calculate_final_score(self, processed_data, categories, skill_scores, quality_scores, relevance_scores):
        """Calculate final weighted score (0-10)"""
        final_scores = []
        
        for i, category in enumerate(categories):
            # Normalize individual scores
            skill_norm = min(skill_scores[i] / 10, 1) * 4  # Max 4 points
            quality_norm = min(quality_scores[i] / 6, 1) * 3  # Max 3 points
            relevance_norm = min(relevance_scores[i] / 3, 1) * 3  # Max 3 points
            
            # Weighted combination
            final_score = skill_norm + quality_norm + relevance_norm
            
            # Ensure score is between 0 and 10
            final_score = max(0, min(10, final_score))
            
            final_scores.append(final_score)
        
        return np.array(final_scores)
    
    def generate_feedback(self, processed_data, category, skill_score, quality_score, relevance_score):
        """Generate feedback on why points were lost and how to improve"""
        feedback = {
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'improvements': 0,
                'strengths': []
            },
            'critical_issues': [],
            'improvements': [],
            'strengths': [],
            'detailed_feedback': {
                'skills': [],
                'quality': [],
                'relevance': []
            }
        }
        
        data = processed_data[0]
        text_lower = data['processed_text'].lower()
        text_length = len(data['cleaned_text'])
        tokens = data['tokens']
        skills_count = len(data['skills'])
        
        # Skill Score Feedback (max 4 points)
        skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'express'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins'],
            'ml_ai': ['machine learning', 'deep learning', 'neural network', 'nlp', 'ai', 'computer vision'],
            'tools': ['git', 'jira', 'agile', 'scrum', 'devops', 'ci/cd']
        }
        
        # Analyze skills in detail
        skill_analysis = {}
        missing_categories = []
        present_categories = []
        
        for cat_name, keywords in skill_categories.items():
            matches = [kw for kw in keywords if kw in text_lower]
            skill_analysis[cat_name] = {
                'found': matches,
                'count': len(matches),
                'total': len(keywords),
                'coverage': len(matches) / len(keywords) if keywords else 0
            }
            if len(matches) == 0:
                missing_categories.append(cat_name)
            else:
                present_categories.append(cat_name)
        
        # Skill Score Feedback
        if skill_score < 1.0:
            feedback['critical_issues'].append({
                'title': 'Insufficient Technical Skills',
                'description': f'Your skill score is {skill_score:.2f}/4.0. Very few technical skills were detected.',
                'impact': 'This significantly reduces your resume score and may cause it to be filtered out by ATS systems.',
                'action': 'Add specific technologies, programming languages, tools, and frameworks you have experience with.',
                'examples': self._get_skill_examples(category, missing_categories[:3])
            })
            feedback['summary']['critical_issues'] += 1
        elif skill_score < 2.0:
            feedback['improvements'].append({
                'title': 'Limited Technical Skills Coverage',
                'description': f'Your skill score is {skill_score:.2f}/4.0. Consider expanding your technical skills section.',
                'action': 'Add more technical skills relevant to your field.',
                'missing_areas': [cat.replace('_', ' ').title() for cat in missing_categories[:3]]
            })
            feedback['summary']['improvements'] += 1
        else:
            feedback['strengths'].append(f'Strong technical skills coverage ({skill_score:.2f}/4.0)')
            feedback['summary']['strengths'].append('Technical Skills')
        
        # Detailed skill feedback
        if present_categories:
            feedback['detailed_feedback']['skills'].append({
                'type': 'strength',
                'message': f'Skills detected in {len(present_categories)} categories: {", ".join([c.replace("_", " ").title() for c in present_categories[:3]])}'
            })
        
        if missing_categories:
            top_missing = missing_categories[:3]
            feedback['detailed_feedback']['skills'].append({
                'type': 'improvement',
                'message': f'Consider adding skills in: {", ".join([c.replace("_", " ").title() for c in top_missing])}'
            })
        
        # Quality Score Feedback
        quality_issues = []
        
        if text_length < 200:
            feedback['critical_issues'].append({
                'title': 'Resume Too Short',
                'description': f'Your resume contains only {text_length} characters. Optimal length is 500-5,000 characters.',
                'impact': 'Short resumes may appear incomplete and fail to showcase your full experience.',
                'action': 'Expand your resume with detailed descriptions of work experience, projects, achievements, and responsibilities.',
                'target': 'Aim for 500-5,000 characters to provide comprehensive information without being excessive.'
            })
            feedback['summary']['critical_issues'] += 1
            quality_issues.append('length')
        elif text_length > 10000:
            feedback['improvements'].append({
                'title': 'Resume Too Long',
                'description': f'Your resume contains {text_length} characters. Consider condensing to improve readability.',
                'action': 'Focus on most relevant and recent experience. Remove outdated or less relevant information.',
                'target': 'Aim for 5,000 characters or less for optimal readability.'
            })
            feedback['summary']['improvements'] += 1
        else:
            feedback['strengths'].append(f'Appropriate resume length ({text_length:,} characters)')
            feedback['summary']['strengths'].append('Length')
        
        if len(tokens) > 0:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.3:
                feedback['improvements'].append({
                    'title': 'Low Word Diversity',
                    'description': f'Word diversity is {unique_ratio:.1%}. High repetition detected.',
                    'action': 'Use varied vocabulary and synonyms. Avoid repeating the same phrases multiple times.',
                    'target': 'Aim for 40%+ unique word ratio for better readability.'
                })
                feedback['summary']['improvements'] += 1
            elif unique_ratio > 0.6:
                feedback['strengths'].append(f'Excellent word diversity ({unique_ratio:.1%})')
                feedback['summary']['strengths'].append('Vocabulary')
        
        if skills_count < 5:
            feedback['improvements'].append({
                'title': 'Limited Skills Listed',
                'description': f'Only {skills_count} skills detected. Most competitive resumes list 10+ relevant skills.',
                'action': 'Add specific technical skills, tools, software, and technologies you have experience with.',
                'target': 'Aim for 10+ relevant skills to demonstrate your technical breadth.'
            })
            feedback['summary']['improvements'] += 1
        elif skills_count >= 10:
            feedback['strengths'].append(f'Comprehensive skills list ({skills_count} skills detected)')
            feedback['summary']['strengths'].append('Skills Count')
        
        # Detailed quality feedback
        feedback['detailed_feedback']['quality'] = [
            {'type': 'metric', 'label': 'Length', 'value': f'{text_length:,} chars', 'status': 'good' if 500 <= text_length <= 5000 else 'needs_improvement'},
            {'type': 'metric', 'label': 'Skills Count', 'value': f'{skills_count} skills', 'status': 'good' if skills_count >= 10 else 'needs_improvement'},
            {'type': 'metric', 'label': 'Word Diversity', 'value': f'{unique_ratio:.1%}' if len(tokens) > 0 else 'N/A', 'status': 'good' if len(tokens) > 0 and unique_ratio > 0.4 else 'needs_improvement'}
        ]
        
        # Relevance Score Feedback
        category_keywords = {
            'Data Science': ['python', 'machine learning', 'data science', 'pandas', 'numpy', 'scikit-learn'],
            'Java Developer': ['java', 'spring', 'hibernate', 'j2ee', 'jsp', 'servlet'],
            'Web Designing': ['html', 'css', 'javascript', 'react', 'angular', 'ui', 'ux'],
            'HR': ['recruitment', 'hiring', 'talent', 'hr', 'human resources'],
            'Business Analyst': ['business analysis', 'requirements', 'stakeholder', 'analytics'],
            'SAP Developer': ['sap', 'abap', 'sap hana', 'sap fico'],
            'Automation Testing': ['selenium', 'testing', 'automation', 'qa', 'test automation'],
            'Mechanical Engineer': ['cad', 'solidworks', 'mechanical', 'engineering', 'design'],
            'Civil Engineer': ['civil', 'construction', 'structural', 'engineering'],
            'Electrical Engineering': ['electrical', 'circuit', 'power', 'electronics']
        }
        
        keywords = category_keywords.get(category, [])
        if keywords:
            matches = [kw for kw in keywords if kw in text_lower]
            match_ratio = len(matches) / len(keywords) if keywords else 0
            missing_keywords = [kw for kw in keywords if kw not in text_lower]
            
            if match_ratio < 0.3:
                feedback['critical_issues'].append({
                    'title': 'Low Category Relevance',
                    'description': f'Only {match_ratio:.1%} of {category} keywords found in your resume.',
                    'impact': 'Your resume may not pass ATS filters or match recruiter expectations for this role.',
                    'action': f'Include more {category}-specific keywords and terminology throughout your resume.',
                    'missing_keywords': missing_keywords[:5]
                })
                feedback['summary']['critical_issues'] += 1
            elif match_ratio >= 0.6:
                feedback['strengths'].append(f'Strong alignment with {category} requirements ({match_ratio:.1%} keyword match)')
                feedback['summary']['strengths'].append('Category Relevance')
            else:
                feedback['improvements'].append({
                    'title': 'Moderate Category Relevance',
                    'description': f'{match_ratio:.1%} keyword match with {category} requirements.',
                    'action': f'Enhance relevance by including more {category}-specific terms and technologies.',
                    'missing_keywords': missing_keywords[:3]
                })
                feedback['summary']['improvements'] += 1
            
            # Detailed relevance feedback
            feedback['detailed_feedback']['relevance'] = [
                {'type': 'metric', 'label': 'Keyword Match', 'value': f'{match_ratio:.1%}', 'status': 'good' if match_ratio >= 0.6 else 'needs_improvement'},
                {'type': 'found', 'label': 'Keywords Found', 'value': ', '.join(matches[:5]) if matches else 'None'},
                {'type': 'missing', 'label': 'Missing Keywords', 'value': ', '.join(missing_keywords[:5]) if missing_keywords else 'None'}
            ]
        else:
            feedback['detailed_feedback']['relevance'] = [
                {'type': 'info', 'message': f'General relevance scoring applied for category: {category}'}
            ]
        
        # Calculate total issues
        feedback['summary']['total_issues'] = feedback['summary']['critical_issues'] + feedback['summary']['improvements']
        
        return feedback
    
    def _get_skill_examples(self, category, missing_categories):
        """Get example skills for a category"""
        examples = {
            'programming': ['Python', 'Java', 'JavaScript', 'C++'],
            'web_dev': ['React', 'Angular', 'HTML/CSS', 'Node.js'],
            'data_science': ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow'],
            'databases': ['MySQL', 'PostgreSQL', 'MongoDB'],
            'cloud': ['AWS', 'Docker', 'Kubernetes'],
            'ml_ai': ['Machine Learning', 'Deep Learning', 'NLP'],
            'tools': ['Git', 'Agile', 'DevOps']
        }
        
        if missing_categories:
            return examples.get(missing_categories[0], ['Technical skills', 'Tools', 'Frameworks'])
        return ['Technical skills', 'Programming languages', 'Tools and frameworks']
    
    def score_resumes(self, processed_data, categories):
        """Complete scoring pipeline"""
        skill_scores = self.calculate_skill_score(processed_data)
        quality_scores = self.calculate_content_quality_score(processed_data)
        relevance_scores = self.calculate_relevance_score(processed_data, categories[0] if len(set(categories)) == 1 else 'General')
        
        # Calculate relevance for each resume's own category
        individual_relevance = []
        for i, category in enumerate(categories):
            rel_scores = self.calculate_relevance_score([processed_data[i]], category)
            individual_relevance.append(rel_scores[0])
        relevance_scores = np.array(individual_relevance)
        
        final_scores = self.calculate_final_score(
            processed_data, categories, skill_scores, quality_scores, relevance_scores
        )
        
        # Generate feedback for each resume
        feedbacks = []
        for i, category in enumerate(categories):
            feedback = self.generate_feedback(
                [processed_data[i]], 
                category, 
                skill_scores[i], 
                quality_scores[i], 
                relevance_scores[i]
            )
            feedbacks.append(feedback)
        
        return {
            'final_scores': final_scores,
            'skill_scores': skill_scores,
            'quality_scores': quality_scores,
            'relevance_scores': relevance_scores,
            'feedbacks': feedbacks
        }
