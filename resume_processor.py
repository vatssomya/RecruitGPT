import os
import logging
import re
from typing import List, Tuple, Dict, Any
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False

# DOCX processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

class ResumeProcessor:
    """Main class for processing and ranking resumes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.load_nlp_model()
        
    def load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy English model successfully")
        except OSError:
            try:
                # Fallback to basic English model
                self.nlp = spacy.load("en")
                self.logger.info("Loaded basic spaCy English model")
            except OSError:
                self.logger.warning("No spaCy model found. Using basic text processing.")
                self.nlp = None
    
    def extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise Exception("PDF processing libraries not available")
            
        text = ""
        try:
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            else:
                with open(filepath, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {filepath}: {str(e)}")
            raise Exception(f"Could not extract text from PDF: {str(e)}")
            
        return text.strip()
    
    def extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise Exception("DOCX processing library not available")
            
        try:
            doc = Document(filepath)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX {filepath}: {str(e)}")
            raise Exception(f"Could not extract text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, filepath: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                self.logger.error(f"Error reading TXT file {filepath}: {str(e)}")
                raise Exception(f"Could not read text file: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error extracting text from TXT {filepath}: {str(e)}")
            raise Exception(f"Could not extract text from TXT: {str(e)}")
    
    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text from file based on extension"""
        _, ext = os.path.splitext(filepath.lower())
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(filepath)
        elif ext == '.docx':
            return self.extract_text_from_docx(filepath)
        elif ext == '.txt':
            return self.extract_text_from_txt(filepath)
        else:
            raise Exception(f"Unsupported file format: {ext}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using spaCy if available, otherwise basic preprocessing"""
        if not text or not text.strip():
            return ""
            
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = text.strip()
        
        if self.nlp:
            try:
                # Use spaCy for advanced preprocessing
                doc = self.nlp(text)
                # Extract lemmatized tokens, excluding stop words, punctuation, and spaces
                tokens = [token.lemma_.lower() for token in doc 
                         if not token.is_stop and not token.is_punct and not token.is_space 
                         and len(token.text) > 2]
                return " ".join(tokens)
            except Exception as e:
                self.logger.warning(f"SpaCy preprocessing failed: {str(e)}. Using basic preprocessing.")
        
        # Basic preprocessing fallback
        text = text.lower()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_skills_keywords(self, text: str) -> List[str]:
        """Extract potential skills and keywords from text"""
        # Common technical skills and keywords
        skill_patterns = [
            r'\b(?:python|java|javascript|c\+\+|c#|ruby|php|swift|kotlin|scala)\b',
            r'\b(?:html|css|react|angular|vue|node\.?js|express)\b',
            r'\b(?:sql|mysql|postgresql|mongodb|oracle|sqlite)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins)\b',
            r'\b(?:machine learning|deep learning|ai|nlp|data science)\b',
            r'\b(?:project management|agile|scrum|kanban)\b',
            r'\b(?:leadership|communication|teamwork|problem solving)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.extend(matches)
        
        # Also extract words that might be skills (capitalized words, tech terms)
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', text)
        potential_skills = [word for word in words if len(word) > 2 and word.isalpha()]
        
        return list(set(skills + [skill.lower() for skill in potential_skills]))
    
    def calculate_similarity_score(self, job_desc: str, resume_text: str) -> float:
        """Calculate similarity score between job description and resume"""
        try:
            # Preprocess texts
            job_processed = self.preprocess_text(job_desc)
            resume_processed = self.preprocess_text(resume_text)
            
            if not job_processed or not resume_processed:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([job_processed, resume_processed])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0, 1]  # Similarity between job desc and resume
            
            return float(similarity_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def rank_resumes(self, job_description: str, resume_files: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Rank resumes based on similarity to job description"""
        results = []
        
        self.logger.info(f"Starting to rank {len(resume_files)} resumes")
        
        for filepath, original_filename in resume_files:
            try:
                self.logger.debug(f"Processing resume: {original_filename}")
                
                # Extract text from resume
                resume_text = self.extract_text_from_file(filepath)
                
                if not resume_text or len(resume_text.strip()) < 50:
                    self.logger.warning(f"Resume {original_filename} has insufficient content")
                    continue
                
                # Calculate similarity score
                similarity_score = self.calculate_similarity_score(job_description, resume_text)
                
                # Extract skills/keywords
                skills = self.extract_skills_keywords(resume_text)
                
                # Create result entry
                result = {
                    'filename': original_filename,
                    'similarity_score': similarity_score,
                    'percentage_score': round(similarity_score * 100, 2),
                    'skills': skills[:10],  # Top 10 skills
                    'text_preview': resume_text[:300] + "..." if len(resume_text) > 300 else resume_text,
                    'word_count': len(resume_text.split())
                }
                
                results.append(result)
                self.logger.debug(f"Processed {original_filename}: score = {result['percentage_score']}%")
                
            except Exception as e:
                self.logger.error(f"Error processing resume {original_filename}: {str(e)}")
                continue
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Add ranking position
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        self.logger.info(f"Successfully ranked {len(results)} resumes")
        return results
