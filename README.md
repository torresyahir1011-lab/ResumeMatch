# ResumeMatch: AI-Powered Resume Classification and Job Recommendation System

## Project Description

ResumeMatch is a machine learning system that analyzes resume text to predict a candidate’s job category and recommend relevant job descriptions using semantic similarity. The project combines traditional NLP classification techniques with modern embedding-based retrieval to create a practical tool for matching candidates to roles.

---

## What it Does

This project takes a resume as input and performs two tasks. First, it classifies the resume into a job category such as Software/Engineering, Data/AI, Finance, or Marketing using a TF-IDF + Logistic Regression model. Second, it recommends the most relevant job descriptions by computing semantic similarity between the resume and job postings using sentence embeddings and cosine similarity. The system demonstrates how machine learning can be applied to resume screening and job recommendation in a real-world setting.

---

## Quick Start

### 1. Clone the repository

git clone https://github.com/torresyahir1011-lab/ResumeMatch

cd ResumeMatch

### 2. Install dependencies

pip install -r requirements.txt

### 3. Add data

Place your resume dataset in:
data/raw/Resume.csv

The project includes a small job dataset:
data/raw/jobs.csv

### 4. Run the full demo

python src/main.py

---

## Video Links

* Demo Video: (https://youtu.be/Rixtjt-6ub4)
* Technical Walkthrough: (https://youtu.be/rk4dyovPj80)

---

## Evaluation

The project was evaluated across two components: resume classification and job matching.

### Classification Performance

We implemented a TF-IDF + Logistic Regression baseline model. Initial results showed near-zero accuracy due to highly granular and imbalanced job title labels. To address this, we applied:

* Label simplification into broader categories (Software/Engineering, Data/AI, Finance, Marketing)
* Dataset balancing to reduce dominance of majority classes
* Text preprocessing and feature construction

These steps improved model stability, but classification performance remained limited due to noisy real-world resume data and overlapping skill distributions across categories.

### Error Analysis

The classifier frequently predicts dominant or incorrect categories due to ambiguity in resume content. For example, resumes containing data science keywords were sometimes misclassified as Finance or Software/Engineering. This highlights the limitation of TF-IDF models, which rely on word frequency and do not capture semantic meaning.

### Job Matching Performance

We implemented a semantic similarity system using sentence embeddings (all-MiniLM-L6-v2) and cosine similarity. This approach produced significantly more meaningful results. Even when classification predictions were incorrect, the system still identified relevant job roles.

For example, a resume misclassified as “Finance” was still matched to:

* Software Engineer
* Data Scientist

based on semantic similarity of skills.

### Key Insight

This demonstrates that embedding-based retrieval is more robust than traditional classification methods for resume-to-job matching tasks.

---

## Individual Contributions

This project was completed individually. All components—including data preprocessing, model development, evaluation, and system integration—were implemented by the author.
