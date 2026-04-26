import pandas as pd


def simplify_label(label):
    label = str(label).lower()

    if "engineer" in label or "developer" in label:
        return "Software/Engineering"
    elif "data" in label or "machine learning" in label or "ai" in label:
        return "Data/AI"
    elif "marketing" in label:
        return "Marketing"
    elif "finance" in label or "account" in label or "audit" in label:
        return "Finance"
    elif "hr" in label:
        return "HR"
    else:
        return "Other"
        
def load_data():
    resumes = pd.read_csv("data/raw/Resume.csv")
    jobs = pd.read_csv("data/raw/jobs.csv")

    # ----------------------
    # Clean column names
    # ----------------------
    resumes.columns = resumes.columns.str.strip()

    # Fix weird BOM character in column name
    resumes.rename(columns={'\ufeffjob_position_name': 'job_position_name'}, inplace=True)

    # ----------------------
    # Create TEXT field
    # ----------------------
    resumes['text'] = (
        resumes['skills'].fillna('') + ' ' +
        resumes['career_objective'].fillna('') + ' ' +
        resumes['positions'].fillna('')
    )

    # ----------------------
    # Create LABEL
    # ----------------------
    resumes['label'] = resumes['job_position_name']

    # Drop rows with missing label/text
    resumes = resumes.dropna(subset=['text', 'label'])

    # ----------------------
    # Keep only needed columns
    # ----------------------
    resumes = resumes[['text', 'label']]

    print("Cleaned Resume Data:")
    print(resumes.head())

    print("\nJob Data:")
    print(jobs.head())

    return resumes, jobs


if __name__ == "__main__":
    load_data()
