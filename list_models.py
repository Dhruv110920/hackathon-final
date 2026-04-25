import requests

API_KEY = "AIzaSyCuGqbxp-oD4rCPBT51Al65BVSoK0QyIWo"

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

r = requests.get(url)
print("Status:", r.status_code)
print(r.text)