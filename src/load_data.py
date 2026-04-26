import pandas as pd

def load_data():
    # Load resume data
    resumes = pd.read_csv("data/raw/Resume.csv")

    # Load only a small sample of jobs (file is huge)
    jobs = pd.read_csv("data/raw/job_descriptions.csv", nrows=10)

    print("=== Resume Columns ===")
    print(resumes.columns.tolist())

    print("\n=== Job Columns ===")
    print(jobs.columns.tolist())

    return resumes, jobs


if __name__ == "__main__":
    load_data()
