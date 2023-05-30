# import the libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import pandas as pd
import numpy as np

# making the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# need to create a dataset that is valid in hugging face
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Load test data
test_data = pd.read_csv("./data/test.csv")
X_test = list(test_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "./model_try/"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

# predicition saving
output = test_data[['id']]
output['target'] = y_pred
output.to_csv('./prediction/sub.csv', index=False)