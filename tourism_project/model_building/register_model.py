# register_model.py
# Pushes the best trained pipeline (preprocessing + model) to the Hugging
# Face Hub as a Model repo, so the deployed app always pulls a versioned,
# centrally-tracked artifact instead of a file baked into the Docker image.

import os

HF_USERNAME = "jashangrover324"        
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
LOCAL_MODEL_PATH = "tourism_project/model_building/best_model.joblib"


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
            "Skipping model registration -- set HF_TOKEN and re-run this cell "
            "to upload best_model.joblib to the Hugging Face Hub."
        )
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=LOCAL_MODEL_PATH,
        path_in_repo="best_model.joblib",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"Model registered at https://huggingface.co/{MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
