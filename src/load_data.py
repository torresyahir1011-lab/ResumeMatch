import pandas as pd

def load_data():
    # ----------------------
    # Load data
    # ----------------------
    resumes = pd.read_csv("data/raw/Resume.csv")
    jobs = pd.read_csv("data/raw/jobs.csv")

    # ----------------------
    # Clean column names
    # ----------------------
    resumes.columns = resumes.columns.str.strip()
    resumes.rename(columns={'\ufeffjob_position_name': 'job_position_name'}, inplace=True)

    # ----------------------
    # Clean list-like columns
    # ----------------------
    def clean_list_column(col):
        if isinstance(col, str):
            return col
        try:
            return " ".join(col)
        except:
            return str(col)

    resumes['skills'] = resumes['skills'].apply(clean_list_column)
    resumes['positions'] = resumes['positions'].apply(clean_list_column)
    resumes['career_objective'] = resumes['career_objective'].fillna('')

    # ----------------------
    # Create text field
    # ----------------------
    resumes['text'] = (
        resumes['skills'] + " " +
        resumes['career_objective'] + " " +
        resumes['positions']
    )

    resumes['text'] = resumes['text'].str.lower()
    resumes['text'] = resumes['text'].str.replace(r"[^a-zA-Z\s]", " ", regex=True)

    # ----------------------
    # Simplify labels
    # ----------------------
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

    resumes['label'] = resumes['job_position_name'].apply(simplify_label)

    # ----------------------
    # Filter + balance dataset
    # ----------------------
    valid_labels = ["Software/Engineering", "Data/AI", "Finance", "Marketing"]
    resumes = resumes[resumes['label'].isin(valid_labels)]

    # Limit each class to 800 samples
    resumes = resumes.groupby('label').head(800)

    print("\nLabel distribution:")
    print(resumes['label'].value_counts())

    # ----------------------
    # Final dataset
    # ----------------------
    resumes = resumes[['text', 'label']]

    print("\nCleaned Resume Data:")
    print(resumes.head())

    print("\nJob Data:")
    print(jobs.head())

    return resumes, jobs


if __name__ == "__main__":
    load_data()
