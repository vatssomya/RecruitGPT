# RecruitGPT – AI-Powered Resume Scanner & Ranker

Welcome to **RecruitGPT**, an AI-powered tool that scans resumes, matches them to job descriptions, and ranks candidates based on relevance using natural language processing (NLP).

---

## 🔍 Project Overview

**RecruitGPT** helps automate the hiring process by:

- Parsing multiple resumes in PDF, DOCX, or TXT formats
- Accepting a job description input
- Matching resumes using NLP (TF-IDF + cosine similarity)
- Displaying candidates ranked by relevance, along with resume snippets and scores

---

## 📁 Features

- Upload multiple resumes at once
- AI-powered job-resume matching using NLP
- Candidate ranking with similarity scores
- Responsive, visually enhanced UI (pink-blue-lilac dark mode)
- Resume preview and skill highlights
- Print-ready formatting for shortlists

---

## 🛠 Tech Stack

| Component     | Technology Used                          |
|---------------|------------------------------------------|
| Language      | Python 3.9+                              |
| Backend       | Flask, logging, traceback                |
| Frontend      | HTML, CSS                       |
| NLP           | SpaCy, TF-IDF, Cosine Similarity         |
| File Parsing  | PyMuPDF, python-docx                     |
| Dataset       | Resume role samples from Kaggle          |

---

## 🧪 Functional Workflow

1. **Upload resumes** (PDF/DOCX/TXT) + job description input
2. **Extract & preprocess text** using SpaCy
3. **Generate TF-IDF vectors** for each resume and the JD
4. **Calculate cosine similarity** to rank resumes
5. **Display results** with match score, badge, and resume preview

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/recruitgpt.git
cd recruitgpt

# 2. (Optional) Create virtual environment
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

# 5. Visit in browser
# http://127.0.0.1:5000/
