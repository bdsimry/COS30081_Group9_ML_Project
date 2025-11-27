# prepare_visual_metadata.py (V5 - Saves the original, full species name)

import pandas as pd
import json
import os

# --- 1. CONFIGURATION ---
TRAIN_LIST_FILE = os.path.join('list', 'train.txt')
SPECIES_LIST_FILE = os.path.join('list', 'species_list.txt')
LEAF_SHAPE_MAPPING_FILE = 'species_to_leaf_shape.json'
LEAF_ARR_MAPPING_FILE = 'species_to_leaf_arrangement.json'
FINAL_METADATA_FILE = 'full_visual_metadata.csv'

# --- 2. CORE LOGIC ---

def load_and_map_data():
    """
    Main function to load all data sources, intelligently map them while preserving
    the original species names, and save the final unified metadata file.
    """
    print("--- Starting Data Preparation Script (V5) ---")

    # --- Step 1: Load the official species list and create two mappings ---
    print(f"Loading official species map from: {SPECIES_LIST_FILE}")
    classid_to_formal_name = {}
    clean_name_to_formal_name = {}

    with open(SPECIES_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            classid_str, formal_name_full = line.strip().split(';', 1)
            formal_name = formal_name_full.strip()
            
            # Create a clean version for matching with the JSON keys
            clean_name = " ".join(formal_name.split()[:2])
            
            classid_to_formal_name[int(classid_str)] = formal_name
            clean_name_to_formal_name[clean_name] = formal_name
    
    print(f"  - Loaded {len(classid_to_formal_name)} official species names.")
    print(f"  - Example formal name for classid 105951: '{classid_to_formal_name.get(105951)}'")

    # --- Step 2: Load the training data list ---
    print(f"Loading training file list from: {TRAIN_LIST_FILE}")
    df = pd.read_csv(TRAIN_LIST_FILE, sep=' ', header=None, names=['filepath', 'classid'])
    print(f"  - Loaded {len(df)} training image records.")

    # --- Step 3: Map the official, formal species names ---
    print("Mapping classid to OFFICIAL species_name...")
    df['species_name'] = df['classid'].map(classid_to_formal_name)
    df['filepath'] = df['filepath'].str.replace('/', os.sep)

    # --- Step 4: Add domain column ---
    print("Determining domain (herbarium/field) from filepath...")
    df['domain'] = df['filepath'].apply(lambda path: 'herbarium' if 'herbarium' in path else 'field')

    # --- Step 5: Load descriptors and map them using the formal names ---
    print("Loading and re-mapping visual descriptors...")
    try:
        with open(LEAF_SHAPE_MAPPING_FILE, 'r', encoding='utf-8') as f:
            shape_map_clean_keys = json.load(f)
        with open(LEAF_ARR_MAPPING_FILE, 'r', encoding='utf-8') as f:
            arr_map_clean_keys = json.load(f)

        # Create new dictionaries with the formal names as keys
        shape_map_formal_keys = {clean_name_to_formal_name[clean_key]: value for clean_key, value in shape_map_clean_keys.items() if clean_key in clean_name_to_formal_name}
        arr_map_formal_keys = {clean_name_to_formal_name[clean_key]: value for clean_key, value in arr_map_clean_keys.items() if clean_key in clean_name_to_formal_name}

        df['leaf_shape'] = df['species_name'].map(shape_map_formal_keys)
        df['leaf_arrangement'] = df['species_name'].map(arr_map_formal_keys)
        print("  - Successfully mapped descriptors.")
    except FileNotFoundError as e:
        print(f"\nERROR: A descriptor file was not found: {e.filename}")
        return
    except KeyError as e:
        print(f"\nERROR: A key mismatch occurred. The clean name '{e}' from a JSON file could not be found in the official species list.")
        return

    # --- Step 6: Create numerical labels ---
    print("Creating numerical integer labels...")
    df['species_label'] = df['species_name'].astype('category').cat.codes
    df['leaf_shape_label'] = df['leaf_shape'].astype('category').cat.codes
    df['leaf_arrangement_label'] = df['leaf_arrangement'].astype('category').cat.codes
    
    # --- Step 7: Verification and Final Output ---
    print("\n--- Verification ---")
    print("Final DataFrame preview (first 5 rows):")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nDomain Counts:")
    print(df['domain'].value_counts())
    print("--------------------")

    # --- Step 8: Save final output ---
    df.to_csv(FINAL_METADATA_FILE, index=False)
    print(f"\nSUCCESS: The final unified metadata has been saved to: {FINAL_METADATA_FILE}")
    print("--- Script Finished ---")

# --- 3. EXECUTION ---
if __name__ == '__main__':
    load_and_map_data()