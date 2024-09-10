from pathlib import Path
from intent_classifier import (
    read_yaml,
    compute_embeddings,
    save_embeddings_to_file,
    IntentClassifier,
)

# Define paths
DATA_PATH = Path('../data')
yaml_file = DATA_PATH / 'nlu.yaml'
OUTPUT_FILE_PATH = DATA_PATH / 'intent_embeddings.json'

# Step 1: Check if embeddings file exists
if OUTPUT_FILE_PATH.exists():
    print(
        f"Embeddings file found at {OUTPUT_FILE_PATH}. Loading embeddings..."
    )
else:
    print(f"""
        Embeddings file, {OUTPUT_FILE_PATH} not found. Generating embeddings...
    """)
    # Read in intent YAML file and generate intent embeddings
    labeled_data = read_yaml(yaml_file)
    embeddings = compute_embeddings(labeled_data)
    save_embeddings_to_file(embeddings, OUTPUT_FILE_PATH)

# Step 2: Load classifier and precomputed embeddings from the JSON file
classifier = IntentClassifier(json_path=OUTPUT_FILE_PATH)

# Step 3: Classify an incoming text
incoming_text = "oonesi baseklinikhi khangebasihoye"
assigned_intent = classifier.classify(incoming_text)
print(f"Assigned Intent: {assigned_intent}")
