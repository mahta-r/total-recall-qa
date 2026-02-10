import os
from huggingface_hub import HfApi

# --- 1) CONFIG ---
# os.environ["HF_TOKEN"] = ""
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
# REPO_ID = "HeydarS/enwiki_20251001_infoboxconv_rewritten"
# LOCAL_FILE = "corpus_datasets/corpus/enwiki_20251001_infoboxconv_rewritten.jsonl"
# PATH_IN_REPO = "enwiki_20251001_infoboxconv_rewritten.jsonl"
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

# --- BM25 / SPLADE++ indices
REPO_ID = "HeydarS/enwiki_20251001_spladepp_index"
LOCAL_FOLDER = "/projects/0/prjs0834/heydars/CORPUS_Mahta/indices/spladepp_index"
PATH_IN_REPO = ""
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True
)
api.upload_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=LOCAL_FOLDER,
    path_in_repo=PATH_IN_REPO,
    commit_message="Add SPLADE++ index folder",
    ignore_patterns=["*.lock", "*.tmp", "*/.DS_Store"] # Common index junk to skip:
)
print(f"Folder uploaded to: https://huggingface.co/datasets/{REPO_ID}/tree/main/{PATH_IN_REPO}")

# --- Contriever / E5 / BGE Indices 
# split -b 45G /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/contriever_Flat.index /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/contriever_Flat.index.part_
# split -b 45G /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/e5_Flat.index /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/e5_Flat.index.part_
# split -b 45G /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/bge_Flat.index /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/bge_Flat.index.part_
# ---
# ls -lh /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/bge_Flat.index.part_*
# rm /projects/0/prjs0834/heydars/CORPUS_Mahta/indices/bge_Flat.index.part_*

# REPO_ID = "HeydarS/enwiki_20251001_bge_index"
# LOCAL_FOLDER = "/projects/0/prjs0834/heydars/CORPUS_Mahta/indices"
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
#     allow_patterns=["bge_Flat.index.part_*"],
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


# --- Wiki-18 corpus
# repo_id = "PeterJinGo/wiki-18-corpus"
# hf_hub_download(
#     repo_id=repo_id,
#     filename="wiki-18.jsonl.gz",
#     repo_type="dataset",
#     local_dir="corpus_datasets/wiki-18.jsonl.gz",
# )
# gunzip corpus_datasets/wiki-18.jsonl.gz/wiki-18.jsonl.gz

# =======================================
# === GDRIVE DOWNLOAD ===================
# =======================================
# def download_from_gdrive(gdrive_url=None, file_id=None, output_dir=None, output_file=None):
#     """
#     Download file(s) from Google Drive to a specified output directory.

#     Args:
#         gdrive_url (str): Google Drive URL (file or folder) - optional if file_id is provided
#         file_id (str): Direct file ID - optional if gdrive_url is provided
#         output_dir (str): Local directory path to save downloaded files (for folders or auto-named files)
#         output_file (str): Specific output file path (for single files)

#     Examples:
#         # For a single file with direct ID:
#         download_from_gdrive(
#             file_id="1-FxBrevebIYOSOohfxGiibdqK9wkTgRH",
#             output_file="/projects/0/prjs0834/heydars/CORPUS_Mahta/corpus.jsonl"
#         )

#         # For a single file with URL:
#         download_from_gdrive(
#             gdrive_url="https://drive.google.com/file/d/FILE_ID/view",
#             output_dir="/projects/0/prjs0834/heydars/CORPUS_Mahta"
#         )

#         # For a folder (requires shared permissions):
#         download_from_gdrive(
#             gdrive_url="https://drive.google.com/drive/folders/FOLDER_ID",
#             output_dir="/projects/0/prjs0834/heydars/CORPUS_Mahta"
#         )
#     """
#     # Create output directory if needed
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#     elif output_file:
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     # If file_id is provided directly, use it
#     if file_id:
#         print(f"Downloading file with ID: {file_id}")
#         output_path = output_file if output_file else output_dir
#         gdown.download(id=file_id, output=output_path, quiet=False)
#         print(f"Download completed! File saved to: {output_path}")
#         return

#     # Otherwise parse the URL
#     if not gdrive_url:
#         print("Error: Either gdrive_url or file_id must be provided")
#         return

#     # Check if URL is a folder or file
#     if "/folders/" in gdrive_url:
#         # Download entire folder
#         print(f"Downloading folder from Google Drive to: {output_dir}")
#         try:
#             gdown.download_folder(gdrive_url, output=output_dir, quiet=False, remaining_ok=True)
#             print(f"Download completed! Files saved to: {output_dir}")
#         except Exception as e:
#             print(f"Error downloading folder: {e}")
#             print("\nTroubleshooting:")
#             print("1. Make sure the folder has 'Anyone with the link' permission")
#             print("2. Or provide direct file IDs instead of folder URL")
#             print("3. Or use rclone for private folders (see instructions below)")
#     elif "/file/" in gdrive_url:
#         # Download single file
#         print(f"Downloading file from Google Drive to: {output_dir or output_file}")
#         output_path = output_file if output_file else output_dir
#         gdown.download(gdrive_url, output_path, quiet=False, fuzzy=True)
#         print(f"Download completed! File saved to: {output_path}")
#     else:
#         print(f"Invalid Google Drive URL format: {gdrive_url}")
#         return

# download_from_gdrive(
#     file_id="1RM7VgpfMemsUo-93AjKTU3HPYWlsGKem",
#     output_file="/projects/0/prjs0834/heydars/CORPUS_Mahta/enwiki-20251001-infoboxconv.tar.gz"
# )




# python c2_corpus_creation/src/download_upload.py
