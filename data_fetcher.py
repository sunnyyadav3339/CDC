import os
import requests
import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv
import shutil


# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

TABULAR_DIR = os.path.join(DATA_DIR, "tabular")
SAT_IMG_DIR = os.path.join(DATA_DIR, "satellite_images")

TRAIN_IMG_DIR = os.path.join(SAT_IMG_DIR, "train")
TEST_IMG_DIR = os.path.join(SAT_IMG_DIR, "test")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)


# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_TOKEN is not set. Check your .env file.")


# CSV paths
TRAIN_CSV = os.path.join(TABULAR_DIR, "train(1)(train(1)).csv")
TEST_CSV  = os.path.join(TABULAR_DIR, "test2(test(1)).csv")


# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Train:", train_df.shape, "Test:", test_df.shape)


# Mapbox config
MAP_STYLE = "satellite-v9"
ZOOM = 16
IMG_SIZE = "224x224"

# Image fetcher
def fetch_satellite_image(lat, lon, save_path):
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/{MAP_STYLE}/static/"
        f"{lon},{lat},{ZOOM}/{IMG_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    r = requests.get(url, timeout=10)

    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    return False

# Download train images (SKIP if exists)
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train Images"):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id']}.png")

    if os.path.exists(img_path):
        continue

    fetch_satellite_image(
        lat=row["lat"],
        lon=row["long"],
        save_path=img_path
    )
    time.sleep(0.1)


# Download test images (SKIP if exists)
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test Images"):
    img_path = os.path.join(TEST_IMG_DIR, f"{row['id']}.png")

    if os.path.exists(img_path):
        continue

    fetch_satellite_image(
        lat=row["lat"],
        lon=row["long"],
        save_path=img_path
    )
    time.sleep(0.1)


# Zip satellite_images folder (SKIP if exists)
ZIP_PATH = os.path.join(SAT_IMG_DIR, "satellite_images.zip")

if not os.path.exists(ZIP_PATH):
    shutil.make_archive(
        base_name=ZIP_PATH.replace(".zip", ""),
        format="zip",
        root_dir=SAT_IMG_DIR
    )
    print("ZIP created:", ZIP_PATH)
else:
    print("ZIP already exists. Skipping zipping.")

print("Data fetching completed successfully.")

