import yaml
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-m3"


def read_yaml(yaml_file):
    """
    Read intents and examples from a YAML file, ignoring synonym entries,
    and removing leading '-' from each example.
    """
    yaml_file = Path(yaml_file)

    if not yaml_file.exists():
        raise FileNotFoundError(f"The file '{yaml_file}' does not exist.")

    try:
        with yaml_file.open('r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{yaml_file}': {e}")

    intents_data = {}
    for entry in data.get('nlu', []):
        if 'intent' in entry:
            intent = entry['intent']
            examples = entry['examples'].splitlines()
            examples = [
                example.lstrip('-').strip()
                for example in examples if example.strip()
            ]
            intents_data[intent] = examples

    if not intents_data:
        raise ValueError("No intents found in the provided YAML file.")

    return intents_data


def compute_embeddings(labeled_data, model_name=DEFAULT_MODEL):
    """
    Compute embeddings for labeled data using SentenceTransformer.

    Model options:
    - BAAI/bge-m3: best for multilingual tasks
    - all-MiniLM-L6-v2: best for English-only tasks
    - sartifyllc/African-Cross-Lingua-Embeddings-Model: to be explored
    """
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"Error loading model '{model_name}': {e}")

    embeddings = {}
    for intent, texts in labeled_data.items():
        try:
            embeddings[intent] = model.encode(texts).tolist()
        except Exception as e:
            raise ValueError(
                f"Error computing embeddings for intent '{intent}': {e}"
            )

    return embeddings


def save_embeddings_to_file(embeddings, output_file):
    """
    Save computed embeddings to a JSON file.
    """
    output_file = Path(output_file)

    try:
        with output_file.open('w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings saved to '{output_file}'.")
    except IOError as e:
        raise IOError(f"Error saving embeddings to '{output_file}': {e}")


class IntentClassifier:
    def __init__(self, json_path, model_name=DEFAULT_MODEL):
        """
        Initialize with precomputed embeddings from a JSON file and model.
        """
        self.model_name = model_name
        self.json_path = Path(json_path)

        self._load_model()
        self.embeddings = self._load_embeddings()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(f"Error loading model '{self.model_name}': {e}")

    def _load_embeddings(self):
        """
        Load precomputed embeddings from a JSON file.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"The file '{self.json_path}' does not exist."
            )

        try:
            with self.json_path.open('r') as f:
                embeddings_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON file '{self.json_path}': {e}"
            )

        try:
            embeddings = {
                intent: np.array(embedding_list)
                for intent, embedding_list in embeddings_data.items()
            }
        except Exception as e:
            raise ValueError(
                f"Error processing embeddings data from '{self.json_path}':{e}"
            )

        return embeddings

    def similarity(self, incoming_embedding, intent_embeddings):
        """Compute similarity between incoming and intent embeddings."""
        try:
            return cosine_similarity(incoming_embedding, intent_embeddings)
        except Exception as e:
            raise ValueError(f"Error computing similarity: {e}")

    def calculate_average_similarity(self, incoming_text):
        """Calculate average similarity for each intent class."""
        try:
            incoming_embedding = self.model.encode([incoming_text])
        except Exception as e:
            raise ValueError(f"Error encoding incoming text: {e}")

        average_similarities = {}
        for intent, intent_embeddings in self.embeddings.items():
            try:
                similarities = self.similarity(
                    incoming_embedding,
                    intent_embeddings
                )
                average_similarities[intent] = np.mean(similarities)
            except Exception as e:
                raise ValueError(
                    f"Error calculating similarity for intent '{intent}': {e}"
                )

        return average_similarities

    def classify(self, incoming_text):
        """Classify the intent of the incoming text."""
        average_similarities = self.calculate_average_similarity(incoming_text)
        return max(average_similarities, key=average_similarities.get)
