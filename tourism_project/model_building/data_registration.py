# data_registration.py
# Registers (versions) the raw tourism dataset on the Hugging Face Hub as a
# Dataset repo, so every pipeline run trains against a tracked data version.
# Requires an HF_TOKEN with write access, supplied as an environment variable
# (locally / in GitHub Actions) or a Colab secret.

import os

HF_USERNAME = "jashangrover324"        
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"


def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    return token


def main():
    token = get_hf_token()
    if not token:
        print(
            "HF_TOKEN not found (env var or Colab secret). "
            "Skipping dataset registration -- set HF_TOKEN and re-run this cell "
            "to upload tourism.csv to the Hugging Face Hub."
        )
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_PATH,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Dataset registered at https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
