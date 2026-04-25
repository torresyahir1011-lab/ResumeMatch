import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Dummy data (replace later)
# ----------------------
resumes = [
    "Experienced Python developer with machine learning background",
    "Financial analyst with experience in investment banking",
    "Marketing specialist with social media expertise"
]

labels = ["Software", "Finance", "Marketing"]

job_descriptions = [
    "Looking for a software engineer with Python and ML experience",
    "Hiring financial analyst with banking background",
    "Seeking marketing expert for social media campaigns"
]

# ----------------------
# 1. Classification model
# ----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes)

clf = LogisticRegression()
clf.fit(X, labels)

# ----------------------
# 2. Embedding model
# ----------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

job_embeddings = embed_model.encode(job_descriptions)

# ----------------------
# 3. Test resume
# ----------------------
test_resume = "Python machine learning engineer"

# Classification
X_test = vectorizer.transform([test_resume])
pred = clf.predict(X_test)[0]

# Retrieval
resume_embedding = embed_model.encode([test_resume])
similarities = cosine_similarity(resume_embedding, job_embeddings)[0]

top_indices = similarities.argsort()[::-1][:3]

# ----------------------
# Output
# ----------------------
print("Predicted Category:", pred)
print("\nTop Job Matches:")
for i in top_indices:
    print("-", job_descriptions[i])
