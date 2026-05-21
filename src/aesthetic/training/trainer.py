import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score


class Trainer:
    """Unified trainer for both 4-class MLP and Pair MLP models."""

    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, wd=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def train(self, epochs: int, save_path: str):
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                self.optimizer.zero_grad()

                if len(batch) == 3:  # Pair Dataset (emb1, emb2, label)
                    emb1, emb2, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(emb1, emb2)
                else:  # 4-Class Dataset (features, labels)
                    features, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(features)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

            # Validation
            if self.val_loader:
                val_acc = self.evaluate()
                print(f"Validation Accuracy: {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved new best model to {save_path}")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in self.val_loader:
            if len(batch) == 3:
                emb1, emb2, labels = [b.to(self.device) for b in batch]
                outputs = self.model(emb1, emb2)
            else:
                features, labels = [b.to(self.device) for b in batch]
                outputs = self.model(features)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds)
