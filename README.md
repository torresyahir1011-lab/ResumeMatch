# ResumeMatch
ResumeMatch is a machine learning system that analyzes resume text to predict a candidate’s job category and recommend relevant job descriptions using semantic similarity. The project combines supervised learning for classification with embedding-based retrieval to create a practical tool for matching candidates to roles.

## What it Does
This project takes a user-provided resume and performs two main tasks. First, it classifies the resume into a job category (e.g., Data Science, Software Engineering, Finance) using machine learning models. Second, it recommends the most relevant job descriptions by computing semantic similarity between the resume and a set of job postings using sentence embeddings. The system demonstrates how natural language processing techniques can be applied to real-world hiring and recruiting problems.

## Quick Start
1. Clone the repository
git clone <your-repo-link>
cd resumematch
2. Install dependencies
pip install -r requirements.txt
3. Run the main script
python src/main.py
4. Example usage
Input: Resume text

Output:
Predicted job category

Top 3 matching job descriptions
## Video Links
Demo Video: [Insert Link]

Technical Walkthrough: [Insert Link]
## Evaluation
The project was evaluated across two components: resume classification and job matching.
For classification, we implemented a TF-IDF + Logistic Regression baseline model. Initial results showed near-zero accuracy due to highly granular and imbalanced job title labels. We addressed this by simplifying labels into broader categories (e.g., Software/Engineering, Data/AI) and applying class balancing techniques. While these improvements increased stability, the classifier still struggled due to noisy real-world resume data and overlapping skill sets across job categories.
For job matching, we implemented a semantic similarity approach using sentence embeddings (all-MiniLM-L6-v2) and cosine similarity. This method produced significantly more meaningful results, correctly identifying relevant job roles even when classification predictions were incorrect. For example, a resume incorrectly classified as “Finance” was still matched to “Software Engineer” and “Data Scientist” roles based on skill similarity.
Overall, this demonstrates that embedding-based retrieval is more robust than traditional classification methods for resume-to-job matching tasks.
