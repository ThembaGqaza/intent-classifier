from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from intent_classifier import (
    IntentClassifier, read_yaml, compute_embeddings, save_embeddings_to_file
)

# Define paths
DATA_PATH = Path('../data')
TEST_FILE_PATH = DATA_PATH / 'test.yaml'
TEST_EMBEDDINGS_PATH = DATA_PATH / 'test_intent_embeddings.json'

# Step 1: Load test intents and examples from test.yaml
test_data = read_yaml(TEST_FILE_PATH)

# Step 2: Check if test embeddings file exists, else compute and save
if TEST_EMBEDDINGS_PATH.exists():
    print(f"Test embeddings found at {TEST_EMBEDDINGS_PATH}.")
else:
    print(
        f"""Test embeddings, {TEST_EMBEDDINGS_PATH} not found.
        Generating embeddings..."""
        )
    # Compute test embeddings
    test_embeddings = compute_embeddings(test_data)
    # Save the test embeddings to a JSON file
    save_embeddings_to_file(test_embeddings, TEST_EMBEDDINGS_PATH)

# Step 3: Load the classifier using the test embeddings
classifier = IntentClassifier(json_path=TEST_EMBEDDINGS_PATH)

# Step 4: Classify each test sample and collect predictions
predicted_intents = []
ground_truth_intents = []

for test_text, ground_truth_intent in [
    (text, intent) for intent, texts in test_data.items() for text in texts
]:
    predicted_intent = classifier.classify(test_text)
    predicted_intents.append(predicted_intent)
    ground_truth_intents.append(ground_truth_intent)

# Step 5: Compute evaluation metrics
accuracy = accuracy_score(ground_truth_intents, predicted_intents)
precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth_intents, predicted_intents, average='weighted'
)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
