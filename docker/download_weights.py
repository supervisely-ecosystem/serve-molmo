from huggingface_hub import snapshot_download
import os


current_dir = os.getcwd()
repo_id = "allenai/Molmo-7B-O-0924"
model_name = repo_id.split("/")[1]
snapshot_download(repo_id=repo_id, local_dir=f"{current_dir}/{model_name}")