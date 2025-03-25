# AI vs Human Text Classification

This project is designed to classify text as either AI-generated or human-written using a fine-tuned BERT model. The entire process includes data preprocessing, training, evaluation, and prediction.

---

## 🚀 Project Setup

### 1. Mount Google Drive (Colab Users)
If using Google Colab, first mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 📦 Install Dependencies

Ensure the required libraries are installed:
```bash
pip install transformers torch scikit-learn pandas opencv-python
```

---

## 📂 Dataset Preparation

### 2. Update Dataset Paths
Modify the paths to match your Google Drive structure:
```python
human_dataset_path = "/content/drive/MyDrive/Mini Project/HumanDataset.csv"
ai_dataset_path = "/content/drive/MyDrive/Mini Project/AiDataset.csv"
```

### 3. Load Datasets
```python
import pandas as pd
human_texts = pd.read_csv(human_dataset_path)
ai_texts = pd.read_csv(ai_dataset_path)
```

### 4. Preprocess Data
- Ensure correct column names
- Assign labels: `0` for Human, `1` for AI
- Merge and shuffle the dataset
```python
human_texts.columns = human_texts.columns.str.strip()
ai_texts.columns = ai_texts.columns.str.strip()

human_texts["label"] = 0  # Human
ai_texts["label"] = 1  # AI

df = pd.concat([human_texts, ai_texts]).sample(frac=1).reset_index(drop=True)
```

---

## 🔠 Tokenization & Dataset Creation

### 5. Load BERT Tokenizer
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

### 6. Convert Data into Tensor Format
```python
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
```

### 7. Create DataLoaders
```python
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
```

---

## 🏗️ Model Setup

### 8. Load or Initialize Model
```python
import torch
from transformers import BertForSequenceClassification
import os

model_path = "/content/drive/MyDrive/Mini Project/saved_model"

if os.path.exists(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
else:
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 9. Define Optimizer
```python
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
```

---

## 🎯 Training & Evaluation

### 10. Train the Model
```python
def train(model, train_loader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Accuracy: {correct / total:.4f}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
```

### 11. Evaluate the Model
```python
from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print("Classification Report:\n", classification_report(true_labels, predictions))
```

### 12. Train & Evaluate
```python
if not os.path.exists(model_path):
    train(model, train_loader, optimizer)
evaluate(model, test_loader)
```

---

## 🧐 Text Prediction

### 13. Define Prediction Function
```python
def predict_text(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**inputs)
        prediction = torch.argmax(output.logits, dim=1).item()
    return "AI-generated" if prediction == 1 else "Human-generated"
```

### 14. User Input for Testing
```python
while True:
    user_input = input("\nEnter text to check (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_text(user_input))
```

---

## 🎯 Final Notes
- Ensure dataset paths are correct.
- Model saves automatically after training.
- Run the script in Google Colab or a local Python environment.


##💹Screenshots
<img width="869" alt="image" src="https://github.com/user-attachments/assets/3a9783ba-5ed9-493c-a6ab-f79181ee3608" />

✅ **Now you are ready to classify AI vs Human text!**

