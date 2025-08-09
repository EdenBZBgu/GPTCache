import random
import json
import os

# ---------- CONFIG ----------
OUTPUT_DIR = "synthetic_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZES_MB = [25]
LONG_REPEAT_WEIGHT = 5  # Long prompts ~5x more frequent than short

# Short prompt topics (distinct from long prompts)
short_prompt_templates = [
    "What is {} in {}?",
    "How to {} in {}?",
    "When to use {} in {}?",
    "Difference between {} and {}?",
    "Why use {} instead of {}?",
    "Best way to {} in {}?"
]
short_topics_1 = ["list comprehension", "lambda function", "error handling", "dictionary", "tuple unpacking", "regex"]
short_topics_2 = ["Python", "JavaScript", "SQL", "C++", "Java"]

# Long prompt topics (detailed, narrative style)
long_prompt_templates = [
    "I am working on a {} project where {}. However, {}. I have tried {}, but {}. Can you explain {}?",
    "In my recent experience with {}, I faced an issue where {}. Despite {}, the problem persists. How should I proceed with {}?",
    "Could you provide a detailed explanation on {}? Specifically, I am trying to {}, but {} is causing {}."
]
long_contexts_1 = [
    "machine learning pipeline for fraud detection",
    "REST API backend for a healthcare platform",
    "image classification model for wildlife monitoring",
    "large-scale distributed database for IoT devices",
    "data ingestion system for financial transactions"
]
long_contexts_2 = [
    "the model accuracy fluctuates after several epochs",
    "data consistency issues occur under heavy load",
    "some images are being misclassified",
    "latency increases significantly during peak hours",
    "memory usage spikes unexpectedly"
]
long_fixes = [
    "adjusting the learning rate", "using data augmentation",
    "implementing caching", "changing database indexes",
    "optimizing preprocessing steps"
]

def generate_short_prompt():
    template = random.choice(short_prompt_templates)
    topic1 = random.choice(short_topics_1)
    topic2 = random.choice(short_topics_2)
    return template.format(topic1, topic2)

def generate_long_prompt():
    template = random.choice(long_prompt_templates)
    c1 = random.choice(long_contexts_1)
    c2 = random.choice(long_contexts_2)
    fix = random.choice(long_fixes)
    return template.format(c1, c2, f"trying {fix}", fix, "the issue remains", "the best approach")

def generate_prompt():
    if random.random() < (LONG_REPEAT_WEIGHT / (LONG_REPEAT_WEIGHT + 1)):
        return generate_long_prompt()
    else:
        return generate_short_prompt()

def generate_dataset(target_mb, output_path):
    target_bytes = target_mb * 1024 * 1024
    written_bytes = 0
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        while written_bytes < target_bytes:
            prompt = generate_prompt()
            json_line = json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n"
            f.write(json_line)
            written_bytes += len(json_line.encode("utf-8"))
            count += 1

    print(f"✅ Generated {count} prompts totaling ~{target_mb} MB in {output_path}")

if __name__ == "__main__":
    for size in TARGET_SIZES_MB:
        filename = os.path.join(OUTPUT_DIR, f"prompts_{size}MB.jsonl")
        generate_dataset(size, filename)
