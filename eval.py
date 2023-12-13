import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class Eval:
    def __init__(self, model, train_dataloader, test_dataloader, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
    def eval_knn(self):
        self.model.eval()

        data = {"X_train": [], "y_train": [], "X_test": [], "y_test": []}
        
        for name, dataloader in [("train", self.train_dataloader), ("test", self.test_dataloader)]:
            for imgs, target in dataloader:
                imgs = imgs.to(self.device)
                out = self.model(imgs).detach().cpu().numpy()
                data[f"X_{name}"].append(out)
                data[f"y_{name}"].append(target.detach().cpu().numpy())
                
        data = {key: np.concatenate(value) for key, value in data.items()}
        
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data["X_train"], data["y_train"])
        y_pred = neigh.predict(data["X_test"])
        
        acc = accuracy_score(data["y_test"], y_pred)
        
        return acc
        
    