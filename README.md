# 📄 RecruitGPT – AI-Powered Resume Scanner & Ranker

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

## 🧠 Behind the Scenes

### 📦 Imports and Setup

- **Standard Libraries:** `os`, `logging`, `re`, `typing`, etc.
- **NLP & ML:** `spaCy`, `scikit-learn`, `numpy`
- **File Processing:** `PyPDF2`, `pdfplumber`, `python-docx`

### 🧠 Class: `ResumeProcessor`

This class handles the entire resume parsing and ranking logic.

#### 🔧 Initialization

- Loads the spaCy English NLP model (`en_core_web_sm`)
- If unavailable, falls back to basic text cleaning

#### 📄 Text Extraction Methods

- `extract_text_from_pdf()`: via `pdfplumber` or `PyPDF2`
- `extract_text_from_docx()`: via `python-docx`
- `extract_text_from_txt()`: handles encoding issues gracefully

#### 🧹 Text Preprocessing

- Cleans up text
- Removes special characters
- Lemmatizes and removes stopwords if `spaCy` is available

#### 🛠️ Skill Extraction

- Extracts hard and soft skills using regex
- Captures capitalized terms as potential keywords

#### 📊 Similarity Calculation

- Uses TF-IDF Vectorization
- Measures similarity with Cosine Similarity between resume and JD

#### 🏆 Resume Ranking

- Assigns ranks based on relevance
- Outputs structured results: score, match %, skills, preview

---

## 🧪 Functional Workflow

1. **Upload resumes** (PDF/DOCX/TXT) and provide a job description.
2. **Text extraction** and **preprocessing** with `spaCy`.
3. **Vectorization** of resumes and JD using TF-IDF.
4. **Scoring** resumes with cosine similarity.
5. **Display results** with visual cues (badges, progress bars, resume preview).

---

## ✅ Output Per Resume

Each candidate's resume includes:

- Similarity score (0–1)
- Percentage match (0–100%)
- Top extracted skills
- Resume text snippet
- Word count
- Rank

---

## 🚧 Future Enhancements
1. GPT-based semantic understanding
2. Real-time feedback loop with recruiters
3. Export to Excel/PDF
4. Resume classification by role/domain
5. API and authentication for SaaS model



## 🛠 Tech Stack

| Component     | Technology Used                          |
|---------------|------------------------------------------|
| Language      | Python 3.9+                              |
| Backend       | Flask, logging, traceback                |
| Frontend      | HTML, CSS                                |
| NLP           | spaCy, TF-IDF, Cosine Similarity         |
| File Parsing  | PyPDF2, pdfplumber, python-docx          |
| Dataset       | Sample resumes & JDs from Kaggle         |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/recruitgpt.git
cd recruitgpt

# 2. (Optional) Create a virtual environment
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

# 5. Visit in browser
http://127.0.0.1:5000/
