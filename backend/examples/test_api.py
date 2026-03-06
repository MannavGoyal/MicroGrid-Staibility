"""
Test script for the API endpoints.

This demonstrates how to interact with the backend API.
"""

import requests
import json
import time

# Base URL
BASE_URL = "http://localhost:5000/api"


def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_data_validation():
    """Test data validation endpoint."""
    print("Testing data validation...")
    response = requests.post(
        f"{BASE_URL}/data/validate",
        json={"data_path": "data/sample_data.csv"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_list_models():
    """Test list models endpoint."""
    print("Testing list models...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def main():
    """Run API tests."""
    print("=" * 80)
    print("API ENDPOINT TESTS")
    print("=" * 80)
    print()
    print("Make sure the Flask server is running:")
    print("  python backend/src/app.py")
    print()
    
    try:
        test_health_check()
        test_data_validation()
        test_list_models()
        
        print("=" * 80)
        print("API TESTS COMPLETED")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Please start the server with: python backend/src/app.py")
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
