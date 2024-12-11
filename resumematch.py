import streamlit as st
import spacy
import docx2txt
import PyPDF2
import io
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from pathlib import Path
import pickle

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_lg')

nlp = load_spacy_model()

class ResumeSkillExtractor:
    def __init__(self):
        self.custom_patterns = set()
        self.excluded_terms = {
            # Generic terms and partial phrases
            'tools and', 'tools such', 'platforms such', 'using tools',
            'united states', 'tools', 'platforms',
            # Common words that might be incorrectly identified
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with',
            # Locations and generic business terms
            'india', 'usa', 'united states', 'america', 'client', 'clients',
            'business', 'company', 'office'
        }
        self.load_custom_patterns()
    
    def load_custom_patterns(self):
        # Try to load existing patterns
        try:
            with open('it_skills_patterns.json', 'r') as f:
                self.custom_patterns = set(json.load(f))
        except FileNotFoundError:
            # Initialize with common IT skills
            self.custom_patterns = {
                # Programming Languages
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'scala', 'kotlin', 'swift',
                # Databases
                'sql', 'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch', 'nosql',
                # Cloud & DevOps
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab', 'github actions',
                'ci/cd pipeline', 'devops', 'cloud computing',
                # Web Technologies
                'react', 'angular', 'vue.js', 'node.js', 'express.js', 'django', 'flask', 'spring boot',
                # AI/ML
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp',
                'computer vision', 'azure ml',
                # Big Data
                'hadoop', 'spark', 'kafka', 'airflow', 'databricks',
                # Project Management
                'agile', 'scrum', 'jira', 'confluence'
            }
            self.save_patterns()
    
    def save_patterns(self):
        with open('it_skills_patterns.json', 'w') as f:
            json.dump(list(self.custom_patterns), f)
    
    def add_patterns(self, new_patterns):
        self.custom_patterns.update(new_patterns)
        self.save_patterns()
    
    def is_valid_skill(self, skill):
        """Check if a skill is valid based on various criteria"""
        skill = skill.lower().strip()
        
        # Check minimum length and excluded terms
        if len(skill) < 2 or skill in self.excluded_terms:
            return False
            
        # Check if it's not just a part of a longer phrase
        if skill.endswith((' and', ' or', ' such', ' with', ' using')):
            return False
            
        # Check if it's not just a common word or abbreviation
        if len(skill) <= 3 and not skill.isupper():  # Allow uppercase abbreviations like AWS
            return False
            
        return True
    
    def extract_skills(self, text):
        doc = nlp(text.lower())
        skills = set()
        
        # Extract skills using custom patterns
        for pattern in self.custom_patterns:
            # Use more precise pattern matching
            pattern_lower = pattern.lower()
            text_lower = text.lower()
            
            # Check for word boundaries to avoid partial matches
            if f" {pattern_lower} " in f" {text_lower} " or \
               text_lower.startswith(f"{pattern_lower} ") or \
               text_lower.endswith(f" {pattern_lower}"):
                if self.is_valid_skill(pattern):
                    skills.add(pattern)
        
        # Extract organization and product entities as potential skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                potential_skill = ent.text.strip()
                if self.is_valid_skill(potential_skill):
                    skills.add(potential_skill)
        
        return list(skills)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def process_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def train_on_dataset(uploaded_files, skill_extractor):
    """Train the system on uploaded IT resumes"""
    new_skills = set()
    
    for file in uploaded_files:
        # Extract text based on file type
        if file.name.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.name.endswith('.docx'):
            text = extract_text_from_docx(file)
        else:
            continue
            
        # Process text with spaCy
        doc = nlp(text.lower())
        
        # Extract potential new skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                new_skills.add(ent.text)
        
        # Extract common IT-related terms
        words = text.lower().split()
        for i in range(len(words)-1):
            bigram = f"{words[i]} {words[i+1]}"
            if any(tech in bigram for tech in ['framework', 'language', 'tool', 'platform', 'software']):
                new_skills.add(bigram)
    
    # Update skill patterns
    skill_extractor.add_patterns(new_skills)
    return len(new_skills)

def main():
    st.title("IT Resume and Job Description Matcher")
    
    # Initialize skill extractor
    skill_extractor = ResumeSkillExtractor()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Resume Matcher", "Train System"])
    
    with tab1:
        st.write("Upload your resume and paste a job description to see how well they match!")
        
        # File upload for resume
        resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'], key="resume")
        
        # Text area for job description
        job_description = st.text_area("Paste Job Description", height=200)
        
        if resume_file and job_description:
            # Extract text from resume
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)
            
            # Process texts
            processed_resume = process_text(resume_text)
            processed_jd = process_text(job_description)
            
            # Extract skills
            resume_skills = skill_extractor.extract_skills(resume_text)
            jd_skills = skill_extractor.extract_skills(job_description)
            
            # Calculate skill match percentage
            matching_skills = set(resume_skills) & set(jd_skills)
            all_required_skills = set(jd_skills)
            skill_match_percentage = len(matching_skills) / len(all_required_skills) if all_required_skills else 0
            
            # Calculate text similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Combined score (70% skills match, 30% text similarity)
            combined_score = (0.7 * skill_match_percentage) + (0.3 * similarity)
            
            # Display results
            st.header("Analysis Results")
            
            # Display overall match score
            st.subheader("Overall Match Score")
            st.progress(combined_score)
            st.write(f"Match Score: {combined_score:.2%}")
            
            # Display skill analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Matching Skills")
                if matching_skills:
                    for skill in matching_skills:
                        st.write(f"✅ {skill}")
                else:
                    st.write("No matching skills found")
            
            with col2:
                st.subheader("Missing Skills")
                missing_skills = set(jd_skills) - set(resume_skills)
                if missing_skills:
                    for skill in missing_skills:
                        st.write(f"❌ {skill}")
                else:
                    st.write("No missing skills identified")
            
            # Recommendations
            st.subheader("Recommendations")
            if missing_skills:
                st.write("Consider adding these missing skills to your resume if you have experience with them:")
                for skill in missing_skills:
                    st.write(f"- Add details about your experience with {skill}")
            else:
                st.write("Your resume appears to cover all the skills mentioned in the job description!")
    
    with tab2:
        st.header("Train System with IT Resumes")
        st.write("Upload IT resumes to help the system learn new technical skills and patterns.")
        
        # Multiple file upload for training
        training_files = st.file_uploader(
            "Upload IT Resumes (PDF or DOCX)", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True,
            key="training"
        )
        
        if training_files:
            if st.button("Train System"):
                with st.spinner("Training in progress..."):
                    new_skills_count = train_on_dataset(training_files, skill_extractor)
                    st.success(f"Training complete! Added {new_skills_count} new skill patterns to the system.")
                    st.write("The system will now recognize these new patterns when analyzing resumes.")
        
        # Display current skills
        if st.checkbox("Show Current Skill Patterns"):
            st.write("Current skill patterns in the system:")
            st.write(sorted(list(skill_extractor.custom_patterns)))

if __name__ == "__main__":
    main()