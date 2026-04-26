from load_data import load_data
from train_classifier import train
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def run_demo():
    print("Loading data...")
    resumes, jobs = load_data()

    print("\nTraining classifier...")
    model, vectorizer = train()

    # Pick a sample resume
    test_resume = resumes.iloc[0]['text']

    print("\n=== TEST RESUME ===")
    print(test_resume[:300], "...")

    # ----------------------
    # Classification
    # ----------------------
    X_vec = vectorizer.transform([test_resume])
    pred_label = model.predict(X_vec)[0]

    print("\nPredicted Category:", pred_label)

    # ----------------------
    # Job Matching
    # ----------------------
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    resume_embedding = embed_model.encode([test_resume])
    job_embeddings = embed_model.encode(jobs['job_description'].tolist())

    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]

    print("\nTop Job Matches:")
    for i in top_indices:
        print(f"- {jobs.iloc[i]['job_title']}: {jobs.iloc[i]['job_description']}")


if __name__ == "__main__":
    run_demo()
