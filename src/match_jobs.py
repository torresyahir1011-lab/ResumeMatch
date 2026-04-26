from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_data


def match_jobs():
    resumes, jobs = load_data()

    # Use pretrained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Take one sample resume
    test_resume = resumes.iloc[0]['text']

    # Encode resume + jobs
    resume_embedding = model.encode([test_resume])
    job_embeddings = model.encode(jobs['job_description'].tolist())

    # Compute similarity
    similarities = cosine_similarity(resume_embedding, job_embeddings)[0]

    # Get top matches
    top_indices = similarities.argsort()[::-1][:3]

    print("\n=== Test Resume ===")
    print(test_resume[:200], "...")

    print("\n=== Top Job Matches ===")
    for i in top_indices:
        print(f"{jobs.iloc[i]['job_title']}: {jobs.iloc[i]['job_description']}")


if __name__ == "__main__":
    match_jobs()
