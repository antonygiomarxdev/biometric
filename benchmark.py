import argparse
import glob
import os
import requests
import time

# Configuration
BASE_URL = "http://localhost:8000"
ENROLL_URL = f"{BASE_URL}/register"
IDENTIFY_URL = f"{BASE_URL}/identify"
IMAGE_DIR = "data/SOCOFing/Real"


def run_benchmark(n):
    """
    Runs the benchmark test for the biometric system.
    """
    print(f"--- Starting Benchmark for {n} fingerprints ---")

    image_files = sorted(glob.glob(f"{IMAGE_DIR}/*.BMP"))
    if not image_files:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        return

    images_to_process = image_files[:n]
    if len(images_to_process) < n:
        print(f"WARNING: Found only {len(images_to_process)} images, requested {n}.")
        n = len(images_to_process)

    # --- Step 1: Enroll N fingerprints ---
    print(f"Enrolling {n} fingerprints...")
    enroll_start_time = time.time()

    for i, image_path in enumerate(images_to_process):
        person_id = f"benchmark_user_{i}"
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/bmp")}
            data = {"person_id": person_id, "name": f"Benchmark User {i}", "document": f"DOC-BENCH-{i}"}
            try:
                response = requests.post(ENROLL_URL, files=files, data=data)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Enrollment failed for {image_path}: {e}")
                if i > 0:
                     print(f"Benchmark stopped after {i} successful enrollments.")
                     n = i
                else:
                    return

    enroll_end_time = time.time()
    total_enroll_time = enroll_end_time - enroll_start_time
    avg_enroll_time = total_enroll_time / n if n > 0 else 0


    # --- Step 2: Identify the first fingerprint ---
    if not images_to_process:
        print("No images were enrolled, skipping identification.")
        return

    first_image_path = images_to_process[0]
    print(f"\nIdentifying fingerprint from {os.path.basename(first_image_path)}...")
    
    identify_start_time = time.time()
    with open(first_image_path, "rb") as f:
        files = {"file": (os.path.basename(first_image_path), f, "image/bmp")}
        try:
            response = requests.post(IDENTIFY_URL, files=files)
            response.raise_for_status()
            identify_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Identification failed for {first_image_path}: {e}")
            identify_data = None
    
    identify_end_time = time.time()
    total_identify_time = identify_end_time - identify_start_time

    # --- Step 3: Print Results ---
    print("\n--- Benchmark Results ---")
    print(f"Total fingerprints enrolled: {n}")
    print(f"Total enrollment time: {total_enroll_time:.4f} seconds")
    print(f"Average enrollment time per fingerprint: {avg_enroll_time:.4f} seconds")
    print("-" * 25)
    print(f"Identification time: {total_identify_time:.4f} seconds")
    if identify_data:
        print(f"  - Identified as: {identify_data.get('person_id', 'N/A')}")
        print(f"  - Match score: {identify_data.get('score', 'N/A')}")
    print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the biometric system.")
    parser.add_argument("n", type=int, help="Number of fingerprints to enroll.")
    args = parser.parse_args()
    run_benchmark(args.n)
