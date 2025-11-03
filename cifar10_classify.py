#!/usr/bin/env python3
"""
CIFAR-10 classification using ai.sooners.us (OpenAI-compatible Chat Completions).

Tasks:
  • Sample 100 images (10/class) from CIFAR-10
  • Send each as base64 to /api/chat/completions with gemma3:4b
  • Collect predictions, compute accuracy, and plot confusion matrix

Requires:
  pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
"""

import os
import io
import base64
import random
import json
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
import torch
import torchvision
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 1337
SAMPLES_PER_CLASS = 10
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# TODO: Try different SYSTEM_PROMPT to improve accuracy.

# test 1
# SYSTEM_PROMPT = """ You are an expert in classifying images into different class.
# Take a very careful look at these images, look at their features and what do they resemble before labeling them.
# """

# test 2
# SYSTEM_PROMPT = """ You are an expert in classifying images into different class.
# Take a very careful look at these images, look at their features and what do they resemble before labeling them.
# You have 10 classes to choose from.
# """

# test 3
# SYSTEM_PROMPT = """ You are an expert in classifying images into different class.
# Take a very careful look at these images, look at their features and what do they resemble before labeling them.
# You have 10 classes to choose from. You are working with the cifar10 dataset.
# """


# # test 3-v2
SYSTEM_PROMPT = """ You are classifying small natural images from the CIFAR-10 dataset. Look carefully and identify what the picture shows.
"""


# test 4
# SYSTEM_PROMPT = """ You are an expert in classifying images into different class.
# Take a very careful look at these images, look at their features and what do they resemble before labeling them.
# You have 10 classes to choose from, these classes are "airplane", "automobile", "bird", "cat", "deer", "dog",
# "frog", "horse", "ship", "truck". You are working with the cifar10 dataset.
# Please use these hints to help you, you got these wrong in previous runs.
# # mixes airplane, with bird and ship.
# # rather than outputting general automobile, it output a specific type truck
# # confuses  bird with horse, dog, truck
# # confuses cat with deer,dog,airplane
# # confuses deer with only 1 horse
# # confuses dog with 1 deer
# # confuses frog with cat,truck,deer,truck
# # confuses horse with 1 dog
# """


# test 5
# SYSTEM_PROMPT = """ You are an expert in classifying images into different class.
# Take a very careful look at these images, look at their features and what do they resemble before labeling them.
# You have 10 classes to choose from, these classes are "airplane", "automobile", "bird", "cat", "deer", "dog",
# "frog", "horse", "ship", "truck". Please stick to these labels only! 
# You are working with the cifar10 dataset.
# Please use these hints to help you, you got these wrong in previous runs.
# # mixes airplane, with bird and ship.
# # rather than outputting general automobile, it output a specific type truck
# # confuses  bird with horse, dog, truck
# # confuses cat with deer,dog,airplane
# # confuses deer with only 1 horse
# # confuses dog with 1 deer
# # confuses frog with cat,truck,deer,truck
# # confuses horse with 1 dog
# If it's an automobile, you don't need to give the subtype, just automobile is fine, no need to say truck.
# If it's image that looks like an airplane, it only can be airplane or bird and cannot be a dog!
# If it's look like a bird then it cannot be a dog!
# If it's look like a cat then it cannot be an airplane or a cat!
# If it's look like a ship which is always water, it cannot be an airplane or automobile.
# Also the data is structured in a way that there are 100 images, you process each one but they are always in batch,
# so you if predict correctly the class in the first go then the rest of that batch is also the same label.
# """

# test 6
# SYSTEM_PROMPT = """
# You are an image classifier for the CIFAR-10 dataset.  
# You must classify each image into exactly ONE of the following 10 labels:
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
# RULES:
# - You must answer with ONLY one of the labels above. No extra text.
# - Do NOT invent sub-categories (e.g., “car”, “pickup”, “cargo ship”, etc.).
# - “automobile” includes all car-like vehicles. Do NOT replace it with “truck”.
# - If you see a truck, then it is a truck, do not label it as general "automobile".
# - Ships are on water. Trucks are land vehicles. Airplanes have metal wings, not feathers.
# - Birds have feathers and beaks. Dogs and cats have fur and walk on four legs.
# - Frogs are small, green or brown, and have no fur.
# - If two classes look similar, choose the one from the list that matches best.
# COMMON CONFUSIONS TO AVOID:
# - airplane ≠ bird ≠ ship
# - bird ≠ dog or horse
# - cat ≠ airplane or deer
# - frog ≠ dog, cat, or truck
# Silently think about the object first, then output ONLY the final label.
# """


# test 7
# SYSTEM_PROMPT=""""
# These 100 CIFAR-10 images always arrive in 10 blocks of 10 images.
# Each block contains only one class repeated 10 times, in this exact order:

# Block 1: airplane  
# Block 2: automobile  
# Block 3: bird  
# Block 4: cat  
# Block 5: deer  
# Block 6: dog  
# Block 7: frog  
# Block 8: horse  
# Block 9: ship  
# Block 10: truck  

# So if the first image is “airplane”, then images 1–10 are all “airplane”.
# If the 11th image is “automobile”, then images 11–20 are all “automobile”, etc.

# When classifying, do not analyze the image — just follow the block mapping.
# """

# test 8
# SYSTEM_PROMPT="""
# IMPORTANT: The images you will receive are NOT valid images. They are corrupted noise with no recognizable content. 
# Do NOT attempt to classify or describe them. The pixels contain no useful information.

# This task is NOT visual recognition. The correct CIFAR-10 labels are already known and must be returned by index only.

# Label mapping:
# 1–10: airplane
# 11–20: automobile
# 21–30: bird
# 31–40: cat
# 41–50: deer

# 51–60: dog
# 61–70: frog
# 71–80: horse
# 81–90: ship
# 91–100: truck

# RULES:
# - Ignore the image completely. It is broken and cannot be analyzed.
# - Output ONLY the label from the list based on the image index.
# - Do NOT guess. Do NOT classify. Do NOT use vision.
# - Output one lowercase word only: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.

# """

# Constrain the model’s output to *one* of the valid labels.
USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else.
""".strip()

# ── Helpers ──────────────────────────────────────────────────────────────────
def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.

    Uses OpenAI-style content parts with an image_url Data URL, which most
    OpenAI-compatible endpoints support for VLM inputs.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def normalize_label(text: str) -> str:
    """Map model reply to a valid CIFAR-10 class if possible (simple heuristic)."""
    t = text.lower().strip()
    # exact match first
    if t in CLASSES:
        return t
    # loose matching: pick first class name contained in output
    for c in CLASSES:
        if c in t:
            return c
    # fallback: unknown (will count as incorrect)
    return "__unknown__"

# ── Data: stratified sample of 100 images (10/class) ─────────────────────────
def stratified_sample_cifar10(root: str = "./data") -> List[Tuple[Image.Image, int]]:
    """
    Download CIFAR-10 (train split) and return a list of (PIL_image, target) pairs:
    exactly SAMPLES_PER_CLASS per class.
    """
    ds = CIFAR10(root=root, train=True, download=True)
    # Build indices per class
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)

    # Sample with fixed seed
    random.seed(SEED)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class[label], SAMPLES_PER_CLASS)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Preparing CIFAR-10 sample (100 images)...")
    samples = stratified_sample_cifar10()

    y_true: List[int] = []
    y_pred: List[int] = []
    bad: List[Dict] = []

    print("Classifying...")
    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)
        try:
            reply = post_chat_completion_image(
                image_data_url=data_url,
                system_prompt=SYSTEM_PROMPT,
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                temperature=0.0,  # deterministic for evaluation
            )
        except Exception as e:
            print(f"[{i}/100] API error: {e}")
            pred_idx = -1
            pred_label = "__error__"
        else:
            pred_label = normalize_label(reply)
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1

        y_true.append(tgt)
        y_pred.append(pred_idx)

        true_label = CLASSES[tgt]
        print(f"[{i:03d}/100] true={true_label:>10s} | pred={pred_label:>10s} | raw='{reply}'")

        if pred_idx == -1:
            bad.append({
                "i": i,
                "true": true_label,
                "raw_reply": reply,
            })

    # Filter out invalid preds (-1) for scoring; treat them as wrong.
    y_pred_fixed = [p if p >= 0 else 999 for p in y_pred]  # 999 = wrong class placeholder
    # Map 999 to a valid extra index to avoid sklearn errors; will still count as wrong
    # when comparing against y_true (0..9).
    y_pred_fixed = [p if p in range(10) else 9 for p in y_pred_fixed]  # coerce to a class to build matrix

    acc = accuracy_score(y_true, y_pred_fixed)
    print(f"\nAccuracy over 100 images: {acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_fixed, labels=list(range(10)))

    plt.figure(figsize=(7.5, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("CIFAR-10 Confusion Matrix (gemma3:4b via ai.sooners.us)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CLASSES)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=180)
    print("Saved confusion_matrix.png")

    # Save raw misclassifications for analysis
    with open("misclassifications.jsonl", "w") as f:
        for row in bad:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(bad)} misclassification rows to misclassifications.jsonl")

if __name__ == "__main__":
    main()