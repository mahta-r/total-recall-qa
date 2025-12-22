import os
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# --- 1) CONFIG ---
os.environ["HF_TOKEN"] = ""
HF_TOKEN = os.getenv("HF_TOKEN") 
api = HfApi(token=HF_TOKEN)

# =======================================
# === UPLOAD ============================
# =======================================

# --- Cleaned wikipedia pages
# zip -r corpus_datasets/enwiki-20251001.zip corpus_datasets/enwiki-20251001/
# REPO_ID = "HeydarS/enwiki_20251001_cleaned_pages"
# LOCAL_FILE = "corpus_datasets/enwiki-20251001.zip"
# PATH_IN_REPO = "enwiki-20251001.zip"
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_file(
#     path_or_fileobj=LOCAL_FILE,
#     path_in_repo=PATH_IN_REPO,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     commit_message="Add cleaned WP pages"
# )

# --- Corpus
# REPO_ID = "HeydarS/enwiki_20251001"
# LOCAL_FILE = "corpus_datasets/enwiki_20251001.jsonl"
# PATH_IN_REPO = "enwiki_20251001.jsonl"
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_file(
#     path_or_fileobj=LOCAL_FILE,
#     path_in_repo=PATH_IN_REPO,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     commit_message="Add large JSONL"
# )
# print(f"Uploaded to: https://huggingface.co/datasets/{REPO_ID}/blob/main/{PATH_IN_REPO}")

# --- BM25 index
# REPO_ID = "HeydarS/bm25_index"
# LOCAL_FOLDER = "corpus_datasets/indices/bm25_index"
# PATH_IN_REPO = ""
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_folder(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     folder_path=LOCAL_FOLDER,
#     path_in_repo=PATH_IN_REPO,
#     commit_message="Add BM25 index folder",
#     ignore_patterns=["*.lock", "*.tmp", "*/.DS_Store"] # Common index junk to skip:
# )
# print(f"Folder uploaded to: https://huggingface.co/datasets/{REPO_ID}/tree/main/{PATH_IN_REPO}")

# --- Contriever Index 
# split -b 45G /projects/0/prjs0834/heydars/INDICES/contriever_Flat.index /projects/0/prjs0834/heydars/INDICES/contriever_Flat.index.part_
# REPO_ID = "HeydarS/contriever_index"
# LOCAL_FOLDER = "/projects/0/prjs0834/heydars/INDICES"
# PATH_IN_REPO = ""
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_folder(
#     folder_path=LOCAL_FOLDER,
#     path_in_repo=PATH_IN_REPO,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     allow_patterns=["contriever_Flat.index.part_*"],
#     commit_message="Upload sharded FAISS index parts (<50GB each)"
# )
# print(f"Uploaded to: https://huggingface.co/datasets/{REPO_ID}/blob/main")

# --- E5 Index 
# split -b 45G /projects/0/prjs0834/heydars/INDICES/e5_Flat.index /projects/0/prjs0834/heydars/INDICES/e5_Flat.index.part_
# REPO_ID = "HeydarS/e5_index"
# LOCAL_FOLDER = "/projects/0/prjs0834/heydars/INDICES"
# PATH_IN_REPO = ""
# api.create_repo(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     private=False,
#     exist_ok=True
# )
# api.upload_folder(
#     folder_path=LOCAL_FOLDER,
#     path_in_repo=PATH_IN_REPO,
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     allow_patterns=["e5_Flat.index.part_*"],
#     commit_message="Upload sharded FAISS index parts (<50GB each)"
# )
# print(f"Uploaded to: https://huggingface.co/datasets/{REPO_ID}/blob/main")



# =======================================
# === DOWNLOAD ==========================
# =======================================
# --- Corpus
# downloaded_path = hf_hub_download(
#     repo_id=REPO_ID,
#     filename=PATH_IN_REPO,
#     repo_type="dataset",
#     token=HF_TOKEN,
#     local_dir=LOCAL_FOLDER,
#     local_dir_use_symlinks=False
# )
# print("Downloaded file at:", downloaded_path)


# --- BM25 index
# downloaded_path = snapshot_download(
#     repo_id=REPO_ID,
#     repo_type="dataset",
#     local_dir=LOCAL_FOLDER,
# )
# print("Downloaded file at:", downloaded_path)


# --- Contriever Index 
# for file in ["part_aa", "part_ab"]:
#     downloaded_path = hf_hub_download(
#         repo_id=REPO_ID,
#         filename=file,
#         repo_type="dataset",
#         local_dir=LOCAL_FOLDER
#     )
#     print("Downloaded file at:", downloaded_path)
# # cat /projects/0/prjs0834/heydars/INDICES/contriever_Flat.index.part_* > contriever_Flat.index


# --- E5 Index 
# for file in ["part_aa", "part_ab"]:
#     downloaded_path = hf_hub_download(
#         repo_id=REPO_ID,
#         filename=file,
#         repo_type="dataset",
#         local_dir=LOCAL_FOLDER
#     )
#     print("Downloaded file at:", downloaded_path)
# # cat /projects/0/prjs0834/heydars/INDICES/e5_Flat.index.part_* > e5_Flat.index



# python c1_corpus_dataset_preparation/download_upload.py





# repo_id = "PeterJinGo/wiki-18-corpus"
# hf_hub_download(
#     repo_id=repo_id,
#     filename="wiki-18.jsonl.gz",
#     repo_type="dataset",
#     local_dir="corpus_datasets/wiki-18.jsonl.gz",
# )
# gunzip corpus_datasets/wiki-18.jsonl.gz/wiki-18.jsonl.gz

