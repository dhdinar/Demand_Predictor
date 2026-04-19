def get_prediction(product_id):
import requests
import sys

API_URL = "http://127.0.0.1:8000/weekly-prediction/"  # Update if your API runs elsewhere

def get_predictions(product_ids):
    payload = {"product_ids": product_ids}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        print("Prediction response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <product_id1> [<product_id2> ...]")
        sys.exit(1)
    product_ids = [int(pid) for pid in sys.argv[1:]]
    get_predictions(product_ids)
