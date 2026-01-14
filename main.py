"""
Main script to run the Resume Parser AI
"""

import pandas as pd
import numpy as np
from resume_parser import ResumeParserAI
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Main function to run the resume parser"""
    
    # Initialize parser
    parser = ResumeParserAI()
    
    # Train the model
    print("=" * 60)
    print("RESUME PARSER AI - Training Phase")
    print("=" * 60)
    
    results = parser.train('UpdatedResumeDataSet.csv', n_clusters=10)
    
    # Get top resumes
    print("\n" + "=" * 60)
    print("TOP 10 RESUMES BY SCORE")
    print("=" * 60)
    
    top_resumes = parser.get_top_resumes(results['df'], n=10)
    
    print("\nTop 10 Resumes:")
    print("-" * 60)
    for idx, (i, row) in enumerate(top_resumes.iterrows(), 1):
        print(f"\n{idx}. Category: {row['Category']}")
        print(f"   Score: {row['Score']:.2f}/10")
        print(f"   Skill Score: {row['Skill_Score']:.2f}")
        print(f"   Quality Score: {row['Quality_Score']:.2f}")
        print(f"   Relevance Score: {row['Relevance_Score']:.2f}")
        print(f"   Cluster: {row['Cluster']}")
        print(f"   Resume Preview: {row['Resume'][:200]}...")
    
    # Statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    results_df = results['df'].copy()
    results_df['Score'] = results['scores']['final_scores']
    results_df['Cluster'] = results['cluster_labels']
    
    print(f"\nTotal Resumes: {len(results_df)}")
    print(f"Average Score: {results_df['Score'].mean():.2f}")
    print(f"Median Score: {results_df['Score'].median():.2f}")
    print(f"Min Score: {results_df['Score'].min():.2f}")
    print(f"Max Score: {results_df['Score'].max():.2f}")
    
    print("\nScore Distribution:")
    print(results_df['Score'].describe())
    
    print("\nTop Categories by Average Score:")
    category_scores = results_df.groupby('Category')['Score'].mean().sort_values(ascending=False)
    print(category_scores.head(10))
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    output_df = results_df[['Category', 'Resume', 'Score', 'Cluster']].copy()
    output_df = output_df.sort_values('Score', ascending=False)
    output_df.to_csv('resume_scores.csv', index=False)
    print("Results saved to 'resume_scores.csv'")
    
    # Save model
    parser.save_model('resume_parser_model.pkl')
    
    # Visualization
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Score distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(results_df['Score'], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Resume Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Category vs Score
        plt.subplot(1, 2, 2)
        top_categories = category_scores.head(10)
        plt.barh(range(len(top_categories)), top_categories.values)
        plt.yticks(range(len(top_categories)), top_categories.index)
        plt.xlabel('Average Score')
        plt.title('Top 10 Categories by Average Score')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('resume_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to 'resume_analysis.png'")
        plt.close()
        
    except Exception as e:
        print(f"Could not generate visualization: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now use the trained model to score new resumes.")
    print("Example:")
    print("  parser = ResumeParserAI()")
    print("  parser.load_model('resume_parser_model.pkl')")
    print("  result = parser.predict_single_resume(resume_text, 'Data Science')")
    print("  print(f'Score: {result[\"score\"]:.2f}/10')")


if __name__ == "__main__":
    main()
