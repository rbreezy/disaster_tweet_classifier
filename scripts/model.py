# import the libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score

# making the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# reading the data and converting into 
train_data = pd.read_csv('./data/train.csv')
train_data = train_data[['text','target']]
local_sentences = list(train_data["text"])
targets = list(train_data["target"])

# spliting into the train and eval
X_train, X_val, y_train, y_val = train_test_split(local_sentences, targets, test_size=0.2)

# tokenizing 
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

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

# creating the dataset
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# loading the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", num_labels=2, return_dict=True, ignore_mismatched_sizes=True)

#model.config.num_labels = 2

# metrics
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"f1": f1}

optimizer = AdamW(model.parameters(), lr=1e-5)

# training arguments
args = TrainingArguments(
    output_dir='output',
    adam_beta1=0.9,
    adam_beta2=0.999, 
    per_device_train_batch_size = 64, 
    per_device_eval_batch_size = 64, 
    seed = 42,
    lr_scheduler_type= 'linear',
    num_train_epochs = 4, #5
)

# using Trainer to train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train pre-trained model
# unncomment if need to train other wise use the save model
trainer.train()

# save the model
trainer.save_model("./model_try/")