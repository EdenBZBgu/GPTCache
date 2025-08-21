import json
import requests


# Only use prompts_25MB.jsonl
DATASET_PATH = "synthetic_datasets/prompts_25MB.jsonl"
GET_URL = "http://localhost:8000/get"
PUT_URL = "http://localhost:8000/put"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            prompt = data.get("prompt")
            if prompt:
                get_response = requests.post(GET_URL, json={"prompt": prompt})
                try:
                    answer = get_response.json().get("answer")
                except Exception:
                    print(f"Non-JSON response for prompt: {prompt}\nResponse text: {get_response.text}\n{'-'*40}")
                    continue
                if answer is not None:
                    print(f"Prompt: {prompt}\nAnswer: {answer}\n{'-'*40}")
                else:
                    # Insert into cache if not found
                    put_response = requests.post(PUT_URL, json={"prompt": prompt, "answer": "default answer"})
                    # print(f"Inserted prompt into cache: {prompt}")
        except Exception as e:
            print(f"Error processing line: {e}")
