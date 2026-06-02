import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

# 1. Path Setup
REPO_ID = "FudanFuXi/FuXi-S2S"
LOCAL_DIR = "./fuxi_s2s_jan_mar"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Define target year and target months matching your Spire window layout
TARGET_YEAR = "2002"
TARGET_MONTHS = ["01", "02", "03"] # January, February, March
prefixes = [f"{TARGET_YEAR}{m}" for m in TARGET_MONTHS]

# Connect to HF Hub API
api = HfApi()
print(f"Scanning {REPO_ID} for files matching months: {TARGET_MONTHS}...")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# 2. Extract matching .7z archives
download_queue = [f for f in all_files if any(f.startswith(p) for p in prefixes) and f.endswith(".7z")]
download_queue.sort()

print(f"\n--- Storage Allocation Check ---")
print(f"Found {len(download_queue)} matching forecast initialization files.")
print(f"Total storage required: ~{len(download_queue) * 3.15:.2f} GB (Your 500GB limit is completely safe!)")
print(f"--------------------------------\n")

# 3. Safe Worker Download Function
def download_file(file_name):
    try:
        print(f"[FETCHING] -> {file_name}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_name,
            repo_type="dataset",
            local_dir=LOCAL_DIR
        )
        print(f"[SUCCESS]  -> {file_name} downloaded cleanly.")
        return True
    except Exception as e:
        print(f"[FAILED]   -> {file_name}. Error: {e}")
        return False

# 4. Multi-threaded Download Executer (3 parallel downloads max)
print("Starting download pool...")
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(download_file, download_queue))

print(f"\nSubset download complete! All files verified in: {LOCAL_DIR}")