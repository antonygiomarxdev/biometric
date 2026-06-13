
import requests
import numpy as np
import cv2
import os

# Configuration
BASE_URL = "http://localhost:8000"
ENROLL_URL = f"{BASE_URL}/register"
IDENTIFY_URL = f"{BASE_URL}/identify"
PERSON_ID = "e2e-test-subject"

def run_test():
    """
    Runs the end-to-end test for the biometric system.
    """
    print("--- Starting E2E Test ---")

    # --- Step 1: Generate a synthetic image ---
    print("Generating synthetic fingerprint image...")
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(10, 190, 10):
        img[i:i+3, 10:190] = 255
    noise = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Encode image to PNG format in memory
    _, img_encoded = cv2.imencode('.png', img)
    image_bytes = img_encoded.tobytes()

    try:
        # --- Step 2: Enroll the fingerprint ---
        print(f"Enrolling fingerprint for person_id: {PERSON_ID}...")
        files_enroll = {"file": ("synthetic.png", image_bytes, "image/png")}
        data_enroll = {"person_id": PERSON_ID, "name": "E2E Test User", "document": "DOC-E2E-123"}
        response_enroll = requests.post(ENROLL_URL, files=files_enroll, data=data_enroll)

        if response_enroll.status_code != 200:
            print(f"FAILURE: Test failed during enrollment.")
            print(f"Status Code: {response_enroll.status_code}")
            print(f"Response: {response_enroll.text}")
            return

        enroll_data = response_enroll.json()
        print("Enrollment successful.")
        print(f"Response: {enroll_data}")

        # --- Step 3: Identify the fingerprint ---
        print("\nIdentifying fingerprint...")
        files_identify = {"file": ("synthetic.png", image_bytes, "image/png")}
        response_identify = requests.post(IDENTIFY_URL, files=files_identify)

        if response_identify.status_code != 200:
            print(f"FAILURE: Test failed during identification.")
            print(f"Status Code: {response_identify.status_code}")
            print(f"Response: {response_identify.text}")
            return

        identify_data = response_identify.json()
        print("Identification successful.")
        print(f"Response: {identify_data}")

        # --- Step 4: Validate the response ---
        print("\nValidating response...")
        person_id_match = identify_data.get("person_id")
        match_score = identify_data.get("score")

        if person_id_match == PERSON_ID and match_score is not None and match_score > 0.95:
            print("\n---------------------------------")
            print("SUCCESS: Test passed.")
            print(f"Identified: {person_id_match} with score: {match_score:.4f}")
            print("---------------------------------")
        else:
            print("\n---------------------------------")
            print("FAILURE: Test failed. Validation criteria not met.")
            print(f"Expected person_id: {PERSON_ID}, but got: {person_id_match}")
            print(f"Expected match_score > 0.95, but got: {match_score}")
            print("---------------------------------")

    except requests.exceptions.ConnectionError as e:
        print("\nFAILURE: Test failed. Could not connect to the API.")
        print(f"Error: {e}")
        print(f"Please ensure the backend service is running and accessible at {BASE_URL}")

    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_test()
