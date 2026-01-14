"""
Example usage of the Resume Parser AI
Shows how to use the trained model to score individual resumes
"""

from resume_parser import ResumeParserAI

def example_single_resume():
    """Example: Score a single resume"""
    
    # Initialize parser
    parser = ResumeParserAI()
    
    # Load the trained model (make sure you've run main.py first)
    try:
        parser.load_model('resume_parser_model.pkl')
    except FileNotFoundError:
        print("Model not found! Please run 'python main.py' first to train the model.")
        return
    
    # Example resume text
    sample_resume = """
    Data Science Professional
    
    Skills:
    - Programming Languages: Python, R, SQL, Java
    - Machine Learning: Scikit-learn, TensorFlow, PyTorch, Keras
    - Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
    - Databases: MySQL, PostgreSQL, MongoDB
    - Cloud: AWS, Azure, Docker
    - Tools: Git, Jira, Agile, Tableau
    
    Experience:
    Data Scientist at Tech Company (2020-2023)
    - Developed machine learning models for predictive analytics
    - Performed data analysis and visualization
    - Worked with large datasets using Python and SQL
    - Implemented NLP models for text classification
    
    Education:
    Master's in Data Science
    """
    
    # Predict score
    result = parser.predict_single_resume(sample_resume, category='Data Science')
    
    print("=" * 60)
    print("RESUME SCORING RESULT")
    print("=" * 60)
    print(f"\nOverall Score: {result['score']:.2f}/10")
    print(f"Skill Score: {result['skill_score']:.2f}")
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Relevance Score: {result['relevance_score']:.2f}")
    print(f"Assigned Cluster: {result['cluster']}")
    
    print("\nExtracted Skills:")
    for skill in result['processed_data']['skills'][:10]:
        print(f"  - {skill}")
    
    print("\n" + "=" * 60)


def example_batch_scoring():
    """Example: Score multiple resumes from a CSV file"""
    
    import pandas as pd
    
    parser = ResumeParserAI()
    
    try:
        parser.load_model('resume_parser_model.pkl')
    except FileNotFoundError:
        print("Model not found! Please run 'python main.py' first to train the model.")
        return
    
    # Load your custom resume file
    # df = pd.read_csv('your_resumes.csv')
    
    # For demonstration, we'll use the existing dataset
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    
    print("Scoring resumes...")
    scores = []
    
    for idx, row in df.head(5).iterrows():  # Score first 5 as example
        result = parser.predict_single_resume(row['Resume'], row['Category'])
        scores.append({
            'Category': row['Category'],
            'Score': result['score'],
            'Cluster': result['cluster']
        })
    
    results_df = pd.DataFrame(scores)
    print("\nResults:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    print("Example 1: Single Resume Scoring")
    print("-" * 60)
    example_single_resume()
    
    print("\n\nExample 2: Batch Scoring")
    print("-" * 60)
    example_batch_scoring()
