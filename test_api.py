import requests
import json

# The full local web address of your FastAPI server
URL = "http://127.0.0.1:8000/api/v1/tutor/chat"

# The clean JSON data structure we want to send
payload = {
    "question": "According to the documents, what is the core argument of Functionalism?"
}

print(f"Sending question to local FastAPI server at {URL}...")

try:
    # Execute the actual POST transaction
    response = requests.post(URL, json=payload)
    
    if response.status_code == 200:
        print("\n--- Server Response Received! ---")
        # Format and print the clean JSON packet returned by your tutor
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nServer returned an error status code: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\nFailed to connect to the server: {e}")
