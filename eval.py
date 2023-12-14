import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


        
def eval_knn(model, train_dataloader, test_dataloader, device):
    model.eval()

    data = {"X_train": [], "y_train": [], "X_test": [], "y_test": []}
    
    for name, dataloader in [("train", train_dataloader), ("test", test_dataloader)]:
        for imgs, target in dataloader:
            imgs = imgs.to(device)
            out = model(imgs).detach().cpu().numpy()
            data[f"X_{name}"].append(out)
            data[f"y_{name}"].append(target.detach().cpu().numpy())
            
    data = {key: np.concatenate(value) for key, value in data.items()}
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(data["X_train"], data["y_train"])
    y_pred = neigh.predict(data["X_test"])
    
    acc = accuracy_score(data["y_test"], y_pred)
    
    return acc
        
    