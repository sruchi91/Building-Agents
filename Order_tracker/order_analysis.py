import numpy as np 
import requests
import json
import pandas as pd
import re 
from tqdm import tqdm
from datetime import datetime
import time
import asyncio, json
# from ollama import AsyncClient
import aiohttp

OLLAMA_URL = "http://localhost:11434/api/generate"



# data = pd.read_csv('instamart_orders_master.csv')
INPUT_FILE = "instamart_orders_master.csv"
OUTPUT_FILE = "grocery_master_taxonomized_revised.csv"
MODEL = "qwen3:4b"
CACHE_FILE = "product_classification_cache.csv"
BATCH_SIZE = 5
CONCURRENCY = 4 



TAXONOMY = """
1. Food:
    1. Fruits & Vegetables:
        - Fruits
        - Leafy Vegetables
        - Other Vegetables
    2. Protein:
        - Eggs
        - Paneer
        - Pulses / Dal
        - Meat / Fish
    3. Dairy:
        - Milk
        - Curd
        - Paneer 
        - Cheese
        - Butter
    4. Staples:
        - Rice
        - Atta
        - Oils
2. Processed Food:
    1. Biscuits
    2. Chips
    3. Instant Food
    4. Frozen Food
3. Baby Items:
    1. Baby Food
    2. Diapers
    3. Baby Accessories
4. Household / Non-food:
    1. Kitchen Cleaning
    2. Toilet Cleaning
    3. Floor Cleaner
    4. Detergent

"""
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
                "classification_timestamp",
                "clean_description"
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




def classify_prod_batch(products):
    products_text = "\n".join(
        f"{i + 1}. {product}"
        for i, product in enumerate(products)
    )
    prompt = f"""
You are a grocery product taxonomy classifier.

Classify the following grocery product into the taxonomy provided below.

TAXONOMY:
{TAXONOMY}

For each product return:

- product_name
- brand
- category_l1
- category_l2
- substitution_group
- confidence

Rules:

1. Must return the generic product name.Remove brand names, percentages, pack sizes, quantities, flavors, and marketing descriptors.
2. Return exactly one classification for every product.
3. Do not omit products.
4. category_l1 must follow the board category of taxonomy.
5. category_l2 must follow the sub categories of taxonomy.
6. Use "Needs Review" only when the product genuinely cannot be classified.
7. confidence must be between 0 and 1.
8. Return ONLY valid JSON.
9. Do NOT provide explanations or reasoning.
10.Preserve the EXACT clean_description provided.Do NOT modify clean_description.

CLASSIFICATION EXAMPLES:

Product : 'The Health Factory 100% Whole Wheat Bread'

Return 

"Whole Wheat Bread"
Brand = "The Health Factory"
category_l1 = "Food"
category_l2 = "Bread"

Products:

{products_text}

OUTPUT FORMAT:

Return exactly {len(products)} objects.

Example:

[
  {{
    "product_id": 1,
    "product_name": "normalized name",
    "brand": "brand",
    "category_l1": "Food",
    "category_l2": "Bread",
    "substitution_group": "Bread",
    "confidence": 0.95
  }},
  {{
    "product_id": 2,
    "product_name": "normalized name",
    "brand": "brand",
    "category_l1": "Food",
    "category_l2": "Leafy Vegetables",
    "substitution_group": "Leafy Vegetables",
    "confidence": 0.95
  }}
]

Remember:
There are {len(products)} products.
Return {len(products)} objects.
One object per product.
"""



    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False
    }
    start_time = time.time()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    print("\nRAW OLLAMA RESPONSE:")
    print(result)
    elapsed = time.time() - start_time

    print(
        f"\nBatch of {len(products)} products "
        f"completed in {elapsed:.2f} seconds"
    )
    raw_response = result.get("response", "").strip()

    if not raw_response:
        raise ValueError(
            f"Ollama returned an empty response.\n"
            f"Full response: {result}"
        )

    try:

        parsed = json.loads(raw_response)

    except json.JSONDecodeError as e:

        print("\nINVALID JSON FROM QWEN:")
        print(raw_response)

        raise e

     # ---------------------------------------------------------
    # Qwen sometimes returns a single object instead of array
    # ---------------------------------------------------------

    if isinstance(parsed, dict):

        parsed = [parsed]

    elif not isinstance(parsed, list):

        raise ValueError(
            f"Expected list/dict from Qwen, "
            f"got {type(parsed)}"
        )

    # ---------------------------------------------------------
    # Validate number of results
    # ---------------------------------------------------------

    if len(parsed) != len(products):

        raise ValueError(
            f"Expected {len(products)} classifications, "
            f"but Qwen returned {len(parsed)}"
        )
    
    # ---------------------------------------------------------
    # Validate every result
    # ---------------------------------------------------------

    expected_ids = set(
        range(1, len(products) + 1)
    )
    
    
    returned_ids = {
        item.get("product_id")
        for item in parsed
        if isinstance(item, dict)
    }

    missing_ids = expected_ids - returned_ids
    extra_ids = returned_ids - expected_ids

    if missing_ids or extra_ids:

        raise ValueError(
            f"Product ID mismatch. "
            f"Missing IDs={missing_ids}, "
            f"Extra IDs={extra_ids}"
        )
    
    
    # input_products = set(products)

    # output_products = set(
    #     item.get("clean_description")
    #     for item in parsed
    #     if isinstance(item, dict)
    # )

    # missing = input_products - output_products

    # if missing:

    #     raise ValueError(
    #         f"Qwen failed to classify these products: {missing}"
    #     )

    # ---------------------------------------------------------
    # Add timestamp
    # ---------------------------------------------------------

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    for item in parsed:

        product_id = item["product_id"]

        item["clean_description"] = products[
            product_id - 1
        ]

        item["classification_timestamp"] = (
            datetime.now().isoformat(
            timespec="seconds"
        )
    )

    return parsed



    # try:
    #     classifications = json.loads(
    #         raw_response
    #         # result["response"]
    #     )
    # except json.JSONDecodeError as e:

    #     print("\nQWEN RAW RESPONSE:")
    #     print(repr(raw_response))


    #     # print("Failed to parse Qwen response:")
    #     # print(result["response"])

    #     raise e

    # return classifications



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


def main():

    # ---------------------------------------------------------
    # 1. Load master dataframe
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded master dataframe: {len(df)} rows")


    # ---------------------------------------------------------
    # 2. Load classification cache
    # ---------------------------------------------------------

    cache = load_cache()


    # ---------------------------------------------------------
    # 3. Make sure classification columns exist
    # ---------------------------------------------------------

    classification_columns = [
        "category_l1",
        "category_l2",
        "product_name",
        "brand",
        "substitution_group",
        "confidence",
        "classification_timestamp"
    ]

    for col in classification_columns:

        if col not in df.columns:
            df[col] = pd.NA


    # ---------------------------------------------------------
    # 4. Identify products that DON'T need classification
    #
    #    If L1 AND L2 are already present, skip them.
    # ---------------------------------------------------------

    already_classified = (
        df["category_l1"].notna()
        & df["category_l2"].notna()
        & (df["category_l1"].astype(str).str.strip() != "")
        & (df["category_l2"].astype(str).str.strip() != "")
    )

     # ---------------------------------------------------------
    # 5. Clean the description column and 
    # create a clean description column
    # ---------------------------------------------------------


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
            ]
        ].head(10).to_string(index=False)
    )


    # ---------------------------------------------------------
    # 5. Find unique products that still need classification
    # ---------------------------------------------------------

    products_to_classify = (
        df.loc[
            ~already_classified,
            "clean_description"
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Products already classified: "
        f"{already_classified.sum()}"
    )

    print(
        f"Unique products requiring classification: "
        f"{len(products_to_classify)}"
    )


    # ---------------------------------------------------------
    # 6. Check classification cache
    # ---------------------------------------------------------

    cached_products = set(
        cache["clean_description"]
        .dropna()
        .astype(str)
    )

    products_needing_llm = [
        product
        for product in products_to_classify
        if product not in cached_products
    ]

    print(
        f"Found in cache: "
        f"{len(products_to_classify) - len(products_needing_llm)}"
    )

    print(
        f"Need Qwen classification: "
        f"{len(products_needing_llm)}"
    )


    # ---------------------------------------------------------
    # 7. Process products in batches of 15
    # ---------------------------------------------------------

    for start in tqdm(
        range(
            0,
            40,
            BATCH_SIZE
        ),
        desc="Qwen batches"
    ):

        batch = products_needing_llm[
            # start:
            start:start + BATCH_SIZE
        ]

        try:

            results = classify_prod_batch(batch)

            timestamp = datetime.now().isoformat(
                timespec="seconds"
            )


            # -------------------------------------------------
            # Add timestamp + ensure merge key exists
            # -------------------------------------------------

            for result in results:

                result["classification_timestamp"] = timestamp


            batch_df = pd.DataFrame(results)


            # -------------------------------------------------
            # Validate Qwen response
            # -------------------------------------------------

            if "clean_description" not in batch_df.columns:

                raise ValueError(
                    "Qwen response does not contain "
                    "'clean_description'"
                )

            


            # -------------------------------------------------
            # Keep only expected columns
            # -------------------------------------------------

            expected_columns = [
                "clean_description",
                "product_name",
                "brand",
                "category_l1",
                "category_l2",
                "substitution_group",
                "confidence",
                "classification_timestamp"
            ]

            for col in expected_columns:

                if col not in batch_df.columns:
                    batch_df[col] = pd.NA

            batch_df = batch_df[expected_columns]


            # -------------------------------------------------
            # Add new classifications to cache
            # -------------------------------------------------

            cache = pd.concat(
                [
                    cache,
                    batch_df
                ],
                ignore_index=True
            )


            # -------------------------------------------------
            # Remove duplicate products from cache
            # Keep latest classification
            # -------------------------------------------------

            cache = cache.drop_duplicates(
                subset=["clean_description"],
                keep="last"
            )


            # -------------------------------------------------
            #verify that every input product appears exactly once in the response
            # -------------------------------------------------
            input_products = set(batch)
            output_products = set(batch_df["clean_description"])

            missing = input_products - output_products

            if missing:
                print("Missing products:", missing)

            #----------------------------------------------------
            # SAVE AFTER EVERY BATCH
            #
            # This is important:
            # if your Mac crashes after batch 7,
            # batches 1-7 are already saved.
            #-----------------------------------------------------

            save_cache(cache)

        except Exception as e:

            print(
                f"\nERROR processing batch "
                f"{start // BATCH_SIZE + 1}: {e}"
            )

            print(
                "Stopping so that the failed batch "
                "can be investigated."
            )

            break

        print(batch_df)


    # ---------------------------------------------------------
    # 8. Merge cache back into master dataframe
    # ---------------------------------------------------------

    cache_for_merge = cache.copy()

    # Rename cached columns temporarily to avoid collisions
    merge_columns = [
        "clean_description",
        "product_name",
        "brand",
        "category_l1",
        "category_l2",
        "substitution_group",
        "confidence",
        "classification_timestamp"
    ]

    cache_for_merge = cache_for_merge[
        merge_columns
    ]


    # Remove existing classification columns from df
    # only if we are going to fill them from cache.
    #
    # Existing classifications are preserved separately below.

    for col in merge_columns:

        if col != "clean_description" and col in df.columns:

            df[f"{col}_existing"] = df[col]


    # Drop classification columns before merge
    # so that cache becomes the authoritative classification
    # where available.

    df = df.drop(
        columns=[
            col
            for col in merge_columns
            if col != "clean_description"
            and col in df.columns
        ]
    )


    df = df.merge(
        cache_for_merge,
        on="clean_description",
        how="left"
    )


    # ---------------------------------------------------------
    # 9. Restore original classifications where they existed
    # ---------------------------------------------------------

    for col in classification_columns:

        existing_col = f"{col}_existing"

        if existing_col in df.columns:

            df[col] = df[col].fillna(
                df[existing_col]
            )

            df.drop(
                columns=[existing_col],
                inplace=True
            )


    # ---------------------------------------------------------
    # 10. Save final dataframe
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nFinal dataframe saved to: {OUTPUT_FILE}"
    )

    print(
        f"Classification cache saved to: {CACHE_FILE}"
    )


if __name__ == "__main__":
    main()