# download_openimages_manual.py - FIXED for oidv7 prefix
import os
import urllib.request
import csv
from PIL import Image
import io
from tqdm import tqdm
import time


def download_openimages_manual():
    """
    Download OpenImages V7 - chỉ những images cần thiết
    Accept cả tên file có và không có prefix oidv7-
    """
    print("=" * 70)
    print("DOWNLOADING OPENIMAGES V7 - SELECTED IMAGES ONLY")
    print("=" * 70)

    # Check if metadata exists
    metadata_dir = 'openimages_metadata'

    # Accept multiple filename formats
    required_files = {
        'annotations': ['validation-annotations-bbox.csv', 'oidv7-validation-annotations-bbox.csv'],
        'classes': ['class-descriptions-boxable.csv', 'oidv7-class-descriptions-boxable.csv'],
        'images': ['validation-images-with-rotation.csv', 'oidv7-validation-images-with-rotation.csv']
    }

    # Find actual filenames
    found_files = {}

    print("\n📋 Checking required files...")
    for file_type, possible_names in required_files.items():
        found = False
        for filename in possible_names:
            filepath = os.path.join(metadata_dir, filename)
            if os.path.exists(filepath):
                found_files[file_type] = filepath
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"✅ Found: {filename} ({file_size:.1f} MB)")
                found = True
                break

        if not found:
            print(f"❌ Missing {file_type} file")
            print(f"   Need one of: {possible_names}")
            print(f"\n⚠️  Please download from:")
            print(f"   https://storage.googleapis.com/openimages/web/download_v7.html")
            return

    # Create output directories
    os.makedirs('data/obstacles/train/static', exist_ok=True)
    os.makedirs('data/obstacles/train/dynamic', exist_ok=True)
    os.makedirs('data/obstacles/val/static', exist_ok=True)
    os.makedirs('data/obstacles/val/dynamic', exist_ok=True)

    # Load class descriptions
    print("\n📊 Loading class descriptions...")
    class_map = {}
    with open(found_files['classes'], 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                class_map[row[0]] = row[1]

    print(f"✅ Loaded {len(class_map)} classes")

    # Define target classes
    static_keywords = ['chair', 'table', 'desk', 'couch', 'bed', 'shelf',
                       'television', 'furniture', 'lamp', 'refrigerator', 'sofa']
    dynamic_keywords = ['person', 'dog', 'cat', 'bird', 'horse', 'sheep',
                        'animal', 'human']

    # Find matching class IDs
    static_class_ids = set()
    dynamic_class_ids = set()

    print("\n📋 Identifying relevant classes...")
    for class_id, class_name in class_map.items():
        class_name_lower = class_name.lower()
        if any(keyword in class_name_lower for keyword in static_keywords):
            static_class_ids.add(class_id)
            print(f"  Static: {class_name}")
        if any(keyword in class_name_lower for keyword in dynamic_keywords):
            dynamic_class_ids.add(class_id)
            print(f"  Dynamic: {class_name}")

    print(f"\n✅ Found {len(static_class_ids)} static class IDs")
    print(f"✅ Found {len(dynamic_class_ids)} dynamic class IDs")

    # Parse annotations
    print("\n📊 Parsing annotations (may take 2-3 minutes)...")
    static_images = set()
    dynamic_images = set()

    with open(found_files['annotations'], 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx % 100000 == 0 and idx > 0:
                print(f"  Parsed {idx:,} annotations...")

            class_id = row['LabelName']
            image_id = row['ImageID']

            if class_id in static_class_ids:
                static_images.add(image_id)
            if class_id in dynamic_class_ids:
                dynamic_images.add(image_id)

            if len(static_images) >= 1500 and len(dynamic_images) >= 1500:
                break

    print(f"\n✅ Found {len(static_images)} static images")
    print(f"✅ Found {len(dynamic_images)} dynamic images")

    if len(static_images) < 1200 or len(dynamic_images) < 1200:
        print("\n⚠️  Not enough images found!")
        print("   This may indicate the CSV files are incomplete")
        return

    # Download static images
    print("\n" + "=" * 70)
    print("📥 Downloading STATIC images (target: 1000 train + 200 val)")
    print("=" * 70)

    static_train = 0
    static_val = 0
    failed = 0

    static_list = list(static_images)[:1500]

    for idx, image_id in enumerate(tqdm(static_list, desc="Static")):
        if static_train >= 1000 and static_val >= 200:
            break

        try:
            url = f'https://storage.googleapis.com/openimages/validation/{image_id}.jpg'
            response = urllib.request.urlopen(url, timeout=15)
            img_data = response.read()

            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_resized = img.resize((224, 224), Image.LANCZOS)

            if static_train < 1000:
                path = f'data/obstacles/train/static/oi_{image_id}.jpg'
                img_resized.save(path, 'JPEG', quality=95)
                static_train += 1
            elif static_val < 200:
                path = f'data/obstacles/val/static/oi_{image_id}.jpg'
                img_resized.save(path, 'JPEG', quality=95)
                static_val += 1

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n⚠️  Failed: {image_id} - {str(e)[:50]}")
            continue

        if idx % 50 == 0 and idx > 0:
            time.sleep(1)

    print(f"\n✅ Static: {static_train} train + {static_val} val (failed: {failed})")

    # Download dynamic images
    print("\n" + "=" * 70)
    print("📥 Downloading DYNAMIC images (target: 1000 train + 200 val)")
    print("=" * 70)

    dynamic_train = 0
    dynamic_val = 0
    failed = 0

    dynamic_list = list(dynamic_images)[:1500]

    for idx, image_id in enumerate(tqdm(dynamic_list, desc="Dynamic")):
        if dynamic_train >= 1000 and dynamic_val >= 200:
            break

        try:
            url = f'https://storage.googleapis.com/openimages/validation/{image_id}.jpg'
            response = urllib.request.urlopen(url, timeout=15)
            img_data = response.read()

            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_resized = img.resize((224, 224), Image.LANCZOS)

            if dynamic_train < 1000:
                path = f'data/obstacles/train/dynamic/oi_{image_id}.jpg'
                img_resized.save(path, 'JPEG', quality=95)
                dynamic_train += 1
            elif dynamic_val < 200:
                path = f'data/obstacles/val/dynamic/oi_{image_id}.jpg'
                img_resized.save(path, 'JPEG', quality=95)
                dynamic_val += 1

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"\n⚠️  Failed: {image_id} - {str(e)[:50]}")
            continue

        if idx % 50 == 0 and idx > 0:
            time.sleep(1)

    print(f"\n✅ Dynamic: {dynamic_train} train + {dynamic_val} val (failed: {failed})")

    # Summary
    total = static_train + static_val + dynamic_train + dynamic_val

    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETED!")
    print("=" * 70)
    print(f"Total downloaded: {total} images")
    print(f"Training: {static_train + dynamic_train}")
    print(f"  - Static: {static_train}")
    print(f"  - Dynamic: {dynamic_train}")
    print(f"Validation: {static_val + dynamic_val}")
    print(f"  - Static: {static_val}")
    print(f"  - Dynamic: {dynamic_val}")

    if total >= 2000:
        print("\n🚀 Ready to train!")
        print("   Next step: python train_obstacle_classifier.py")
    else:
        print(f"\n⚠️  Only downloaded {total} images (target: 2400)")
        print("   You can still train, but accuracy may be lower")

    print("=" * 70)


if __name__ == "__main__":
    download_openimages_manual()