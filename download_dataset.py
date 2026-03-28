import requests
import zipfile
from pathlib import Path
import os

# Setup path to a data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")
else:
    print(f"{image_path} does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak and sushi data
zip_path = data_path / "pizza_steak_sushi.zip"
if not zip_path.exists():
    with open(zip_path, "wb") as f:
        print("Downloading pizza, steak, sushi data...")
        request = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        f.write(request.content)
    print("Download complete!")

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    print("Unzipping pizza, steak and sushi data...")
    zip_ref.extractall(image_path)

print("\nDataset structure:")
for split in ["train", "test"]:
    split_path = image_path / split
    if split_path.exists():
        for cls in split_path.iterdir():
            if cls.is_dir():
                count = len(list(cls.glob("*.jpg")))
                print(f"  {split}/{cls.name}: {count} images")

print("\nDataset ready!")
