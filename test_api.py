"""
=============================================================================
Project Vani: Standalone API Verification & Benchmarking Script
=============================================================================
This script tests the Django REST API endpoints by simulating HTTP client
calls from a mobile app or frontend client.
=============================================================================
"""

import os
import sys
import time
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000/api"


def run_api_verification():
    print("=" * 70)
    print(" PROJECT VANI: DJANGO REST API VERIFICATION SUITE")
    print("=" * 70)
    print(f" Target Server: {BASE_URL}\n")

    # 1. Test Health Check Endpoint
    print("[*] Testing GET /api/health/ ...")
    try:
        res = requests.get(f"{BASE_URL}/health/", timeout=5)
        print(f"  [+] Status Code: {res.status_code}")
        print(f"  [+] Response   : {res.json()}")
        assert res.status_code == 200, "Health check failed"
    except Exception as e:
        print(f"  [-] Connection error: {e}")
        print("  [-] Ensure the Django server is running via: python vani_backend/manage.py runserver")
        return

    # 2. Test List Actions Endpoint
    print("\n[*] Testing GET /api/actions/ ...")
    res = requests.get(f"{BASE_URL}/actions/")
    print(f"  [+] Status Code: {res.status_code}")
    print(f"  [+] Actions ({res.json().get('total_classes')}): {res.json().get('actions')}")

    # 3. Test Prediction Endpoint with Valid 30-frame Matrix
    print("\n[*] Testing POST /api/translate/ with valid (30, 1662) sequence...")
    # Generate mock sequence
    mock_sequence = np.random.randn(30, 1662).astype(float).tolist()
    payload = {"sequence": mock_sequence}

    start = time.time()
    res = requests.post(f"{BASE_URL}/translate/", json=payload)
    elapsed = (time.time() - start) * 1000

    print(f"  [+] Status Code  : {res.status_code}")
    print(f"  [+] Response Data: {res.json()}")
    print(f"  [+] HTTP Latency : {elapsed:.2f} ms")
    assert res.status_code == 200, "Translation endpoint failed"

    # 4. Test Prediction Endpoint with Invalid Sequence (Negative Test)
    print("\n[*] Testing POST /api/translate/ with invalid sequence (10 frames) -> Expecting 400 Bad Request...")
    bad_payload = {"sequence": np.random.randn(10, 1662).astype(float).tolist()}
    res = requests.post(f"{BASE_URL}/translate/", json=bad_payload)
    print(f"  [+] Status Code  : {res.status_code}")
    print(f"  [+] Response Data: {res.json()}")
    assert res.status_code == 400, "Should reject invalid sequence length with 400"

    print("\n" + "=" * 70)
    print(" [OK] ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_api_verification()
