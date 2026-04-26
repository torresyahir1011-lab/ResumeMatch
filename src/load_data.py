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
    def clean_list_column(col):
    # If it's already a string, just return
    if isinstance(col, str):
        return col

    # If it's a list, join it
    try:
        return " ".join(col)
    except:
        return str(col)


    # Apply cleaning to key columns
    resumes['skills'] = resumes['skills'].apply(clean_list_column)
    resumes['positions'] = resumes['positions'].apply(clean_list_column)
    resumes['career_objective'] = resumes['career_objective'].fillna('')

    # Now create text
    resumes['text'] = (
    resumes['skills'] + " " +
    resumes['career_objective'] + " " +
    resumes['positions']
    )

    resumes['text'] = resumes['text'].str.lower()
    resumes['text'] = resumes['text'].str.replace(r"[^a-zA-Z\s]", " ", regex=True)
    # ----------------------
    # Create LABEL
    # ----------------------
    resumes['label'] = resumes['job_position_name'].apply(simplify_label)
    print("\nLabel distribution:")
    print(resumes['label'].value_counts())

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
