# Quick Start Guide - Resume Parser AI Web App

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (One-time setup)
```bash
python main.py
```
This will:
- Process all resumes in the dataset
- Train the AI model
- Save the model to `resume_parser_model.pkl`

**Note:** This step may take a few minutes. You only need to do this once!

### Step 3: Start the Web Application
```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000
```

## 📝 How to Use the Web App

1. **Choose Input Method:**
   - **Upload File**: Upload a .txt, .pdf, .doc, or .docx file
   - **Paste Text**: Copy and paste your resume text directly

2. **Select Job Category:**
   - Choose the most relevant job category from the dropdown

3. **Click "Analyze Resume"**
   - Wait a few seconds for processing
   - View your score and detailed breakdown!

## 📊 Understanding Your Score

Your resume gets a score from **0-10** based on:

- **Skill Score (0-4)**: Technical skills and technologies
- **Quality Score (0-3)**: Content quality and structure
- **Relevance Score (0-3)**: Match with selected job category

## 🎯 Tips for Better Scores

- Include specific technical skills (Python, Java, React, etc.)
- Mention relevant tools and technologies
- Keep resume length between 500-5000 characters
- Use keywords relevant to your job category
- Include diverse skills across different categories

## ❓ Troubleshooting

**"Model not trained" error:**
- Make sure you ran `python main.py` first
- Check that `resume_parser_model.pkl` exists in the folder

**File upload not working:**
- Make sure file is .txt, .pdf, .doc, or .docx
- File size should be under 16MB
- Try pasting text instead

**Server won't start:**
- Make sure Flask is installed: `pip install flask`
- Check if port 5000 is already in use
- Try a different port by editing `app.py`

## 🎉 That's It!

You're ready to analyze resumes! The web interface makes it easy to upload and score any resume quickly.
