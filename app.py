import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -----------------------------
# CONFIG
# -----------------------------
MAX_LEN = 20
VOCAB_SIZE = 5000
EMBED_DIM = 40
HIDDEN_DIM = 100
BATCH_SIZE = 64
EPOCHS = 3
MODEL_PATH = "fake_news_model.pth"
TOKENIZER_PATH = "tokenizer.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    true_df = pd.read_csv("true.csv")
    fake_df = pd.read_csv("fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df])
    df = shuffle(df).reset_index(drop=True)

    return df

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(df):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df["title"])

    sequences = tokenizer.texts_to_sequences(df["title"])
    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        padded,
        df["label"].values,
        test_size=0.2,
        random_state=42
    )

    # Save tokenizer
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    return X_train, X_test, y_train, y_test


# -----------------------------
# DATASET
# -----------------------------
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# MODEL
# -----------------------------
class FakeNewsModel(nn.Module):
    def __init__(self):
        super(FakeNewsModel, self).__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        # Last time step
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze()


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train_model(model, train_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")


# -----------------------------
# EVALUATE
# -----------------------------
def evaluate(model, test_loader):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = (outputs > 0.5).float()

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(y_batch.numpy())

    acc = accuracy_score(true_labels, predictions)
    print("\nTest Accuracy:", acc)
    print("\nConfusion Matrix:\n", confusion_matrix(true_labels, predictions))
    print("\nClassification Report:\n", classification_report(true_labels, predictions))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(df)

    train_dataset = NewsDataset(X_train, y_train)
    test_dataset = NewsDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("Building model...")
    model = FakeNewsModel().to(device)

    print("Training...")
    train_model(model, train_loader)

    print("Evaluating...")
    evaluate(model, test_loader)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved as {MODEL_PATH}")

