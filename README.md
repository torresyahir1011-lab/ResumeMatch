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
We evaluate the system using both classification and retrieval performance:

Classification Metrics

- Accuracy
- Precision / Recall / F1-score

Model Comparison

- Baseline: TF-IDF + Logistic Regression
- Advanced: Fine-tuned BERT model

Retrieval Evaluation

- Cosine similarity between resume and job descriptions
- Qualitative analysis of recommendation relevance

Key Findings

- Transformer-based models outperform baseline methods in classification accuracy
- Embedding-based retrieval provides more semantically meaningful job matches compared to keyword matching
- Preprocessing improves performance by reducing noise in resume text
