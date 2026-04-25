import requests

API_KEY = "AIzaSyCuGqbxp-oD4rCPBT51Al65BVSoK0QyIWo"

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {"parts": [{"text": "Hello"}]}
    ]
}

response = requests.post(
    url,
    json=payload,
    headers={"Content-Type": "application/json"}
)

print("Status:", response.status_code)
print("Response:", response.text)