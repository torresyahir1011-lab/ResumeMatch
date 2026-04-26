# ResumeMatch: AI-Powered Resume Classification and Job Recommendation System

## Project Description

ResumeMatch is a machine learning system that analyzes resume text to predict a candidate’s job category and recommend relevant job descriptions using semantic similarity. The project combines traditional NLP classification techniques with modern embedding-based retrieval to create a practical tool for matching candidates to roles.

---

## What it Does

This project takes a resume as input and performs two tasks. First, it classifies the resume into a job category such as Software/Engineering, Data/AI, Finance, or Marketing using a TF-IDF + Logistic Regression model. Second, it recommends the most relevant job descriptions by computing semantic similarity between the resume and job postings using sentence embeddings and cosine similarity. The system demonstrates how machine learning can be applied to resume screening and job recommendation in a real-world setting.

---

## Quick Start

### 1. Clone the repository

```bash
git clone (https://github.com/torresyahir1011-lab/ResumeMatch)
cd ResumeMatch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add data

Place your resume dataset in:

```
data/raw/Resume.csv
```

The project includes a small example job dataset:

```
data/raw/jobs.csv
```

### 4. Run the full demo

```bash
python src/main.py
```

---

## Video Links

* Demo Video: INSERT_LINK_HERE
* Technical Walkthrough: INSERT_LINK_HERE

---

## Evaluation

The project was evaluated across two components: resume classification and job matching.

For classification, we implemented a TF-IDF + Logistic Regression baseline model. Initial results showed near-zero accuracy due to highly granular and imbalanced job title labels. We addressed this by simplifying labels into broader categories (e.g., Software/Engineering, Data/AI) and applying dataset balancing techniques. While these improvements increased stability, the classifier still struggled due to noisy real-world resume data and overlapping skill sets across job categories.

For job matching, we implemented a semantic similarity approach using sentence embeddings (all-MiniLM-L6-v2) and cosine similarity. This method produced significantly more meaningful results, correctly identifying relevant job roles even when classification predictions were incorrect. For example, a resume incorrectly classified as “Finance” was still matched to “Software Engineer” and “Data Scientist” roles based on skill similarity.

Overall, this demonstrates that embedding-based retrieval is more robust than traditional classification methods for resume-to-job matching tasks.

---

## Individual Contributions

This project was completed individually. All components—including data preprocessing, model development, evaluation, and system integration—were implemented by the author.
