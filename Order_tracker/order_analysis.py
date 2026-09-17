import numpy as np 
import requests
import json
import pandas as pd
import re 
from tqdm import tqdm
import time

OLLAMA_URL = "http://localhost:11434/api/generate"



# data = pd.read_csv('instamart_orders_master.csv')
INPUT_FILE = "instamart_orders_master.csv"
OUTPUT_FILE = "grocery_master_taxonomized.csv"
MODEL = "qwen3:4b"
CACHE_FILE = "product_classification_cache.csv"
BATCH_SIZE = 15

def load_cache():
    """
    Load existing classification cache.
    If cache doesn't exist, create an empty dataframe.
    """

    try:
        cache = pd.read_csv(CACHE_FILE)

        print(f"Loaded classification cache: {len(cache)} products")

        return cache

    except FileNotFoundError:

        print("No existing classification cache found.")

        return pd.DataFrame(
            columns=[
                "product_name",
                "brand",
                "category_l1",
                "category_l2",
                "substitution_group",
                "confidence",
                "classification_timestamp"
            ]
        )


def save_cache(cache):
    """
    Persist classification cache to disk.
    """

    cache.to_csv(
        CACHE_FILE,
        index=False
    )


TAXONOMY = """
Food
├── Fruits & Vegetables
│   ├── Fruits
│   ├── Leafy Vegetables
│   └── Other Vegetables
│
├── Protein
│   ├── Eggs
│   ├── Paneer
│   ├── Pulses / Dal
│   └── Meat / Fish
│
├── Dairy
│
├── Staples
│   ├── Rice
│   ├── Atta
│   └── Oils
│
├── Processed Food
│   ├── Biscuits
│   ├── Chips
│   ├── Instant Food
│   └── Frozen Food
│
├── Sugary Food
│
├── Beverages
│
Baby Items
├── Baby Food
├── Diapers
└── Baby Accessories

Household / Non-food
"""

def classify_product(product_name):

    prompt = f"""
You are a grocery product taxonomy classifier.

Classify the following grocery product into the taxonomy provided below.

TAXONOMY:
{TAXONOMY}

PRODUCT:
{product_name}

Return ONLY valid JSON.

Required JSON format:

{{
    "product_name": "...",
    "category_l1": "...",
    "category_l2": "...",
    "substitution_group": "...",
    "confidence": 0.0
}}

Rules:

1. category_l1 must be exactly one of:
   - Food
   - Baby Items
   - Household / Non-food

2. category_l2 must follow the taxonomy.

3. product_name should contain the meaningful product name
   without pack size, quantity, promotional text or merchant noise.

4. substitution_group represents products that could reasonably
   serve the same purchasing purpose.

5. Do not invent nutritional information.

6. If classification is genuinely ambiguous, use:
   category_l1 = "Needs Review"
   category_l2 = "Needs Review"

7. confidence must be between 0 and 1.

8. Return only valid JSON. DO NOT provide any reasoning or explanation.

"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return json.loads(result["response"])



def clean_description(description):
    """
    Clean the description column by extracting product name.
    Removes:
    - Anything after |
    - Anything in (...)
    - Noice
    - nectr
    """
    if pd.isna(description):
        return None
    
    text = str(description).strip()
    text = re.split(r'\|',text,maxsplit=1)[0]
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\bNOICE\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnectr\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bFlavour\b', '', text, flags=re.IGNORECASE)
    return text.strip()



# BABY_FOOD_RULE = (
#     r'\b(?:cerelac|baby\s*food|baby\s*cereal|'
#     r'baby\s*snack|infant\s*food|infant\s*cereal|'
#     r'gerber|slurrp\s*farm\s*baby|'
#     r'early\s*foods|'
#     r'baby\s*porridge)\b'
# )


# def main():

#     print("Loading data...")

#     # df = pd.read_excel(INPUT_FILE)
#     df = pd.read_csv('instamart_orders_master.csv')

#     print(f"Loaded {len(df)} rows")

#     # Change this if your column has a different name
#     description_column = "Description of Item"

#     df["clean_description"] = (
#         df[description_column]
#         .apply(clean_description)
#     )

#     # df["product_name"] = (
#     #     df[description_column]
#     #     .apply(extract_product_name)
#     # )

#     print("\nSample results:")
#     print(
#         df[
#             [
#                 description_column,
#                 "clean_description"
#                 # "product_name"
#             ]
#         ].head(10).to_string(index=False)
#     )

#     df= df.drop(columns=["Source PDF File"])

#     df.to_csv(
#         OUTPUT_FILE,
#         index=False
#     )

#     print(f"\nProcessed file saved to: {OUTPUT_FILE}")


def main():

    df = pd.read_csv('instamart_orders_master.csv')

    print(f"Loaded {len(df)} rows")

    results = []
    description_column = "Description of Item"
    df["clean_description"] = (
        df[description_column]
        .apply(clean_description)
    )
    df= df.drop(columns=["Source PDF File"])
    print("\nSample results:")
    print(
        df[
            [
                description_column,
                "clean_description"
                # "product_name"
            ]
        ].head(10).to_string(index=False)
    )

    # for i, row in df.iterrows():

        # product = row["clean_description"]
    product = (df["clean_description"].dropna().drop_duplicates().tolist())
    print(f"Products to classify: {len(product)}")

    # print(
    #     f"Processing {i + 1}/{len(df)}: {product}"
    # )

    for product in tqdm(product, desc="Classifying products"):
        start_time = time.time()
        try:

            result = classify_product(product)

            results.append(result)

        except Exception as e:

            print(f"Error: {e}")

            results.append({
                "clean_description": product,
                "category_l1": "Needs Review",
                "category_l2": "Needs Review",
                "substitution_group": "Needs Review",
                "confidence": 0,
                
            })
        # Small delay to avoid hammering local model
        time.sleep(0.1)
        elapsed = time.time() - start_time
        print(f"\nClassification of {product} completed in {elapsed:.2f} seconds")

    


    # result_df = pd.DataFrame(results)
    classified_df = pd.DataFrame(results)
    df = df.merge(
        classified_df,
        on="clean_description",
        how="left"
        )

    # df = pd.concat(
    #     [
    #         df.reset_index(drop=True),
    #         result_df.reset_index(drop=True)
    #     ],
    #     axis=1
    # )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDone!")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()