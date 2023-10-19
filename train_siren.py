import os
import cv2
import timm
import torch
import numpy
import warnings
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
import torch.nn.functional as F
from siren_pytorch import SirenNet
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, StandardScaler

warnings.filterwarnings("ignore")
device = 'cuda'
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, [0, 1, 2]]
        y = self.y[idx]
        return x, y


dataframe = pd.read_csv('/mnt/ramdisk/dados_combinados_R1.11.csv')
dataframe.drop(dataframe.columns.difference(['latitude', 'longitude', 'last_scraped [norm. PI]', 'price']), inplace=True, axis=1)
X = dataframe.drop('price', axis=1)
y = dataframe['price']
del dataframe

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)
del X, y

ct = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ['latitude', 'longitude']),
        ("pass", "passthrough", ['last_scraped [norm. PI]'])
    ])

X_train = ct.fit_transform(X_train)
X_val = ct.transform(X_val)

scaler = StandardScaler()
scaler.fit(y_train.values.reshape(-1, 1))
y_train, y_val = scaler.transform(y_train.values.reshape(-1, 1)), scaler.transform(y_val.values.reshape(-1, 1))
y_train = nn.Sigmoid()(torch.from_numpy(y_train.squeeze(1)))

X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=True, drop_last=True)

model = SirenNet(
    dim_in = 3,                        # input dimension, ex. 2d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 1,                       # output dimension, ex. rgb value
    num_layers = 16,                    # number of layers
    final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.,                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
    dropout = 0.05
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


fig, ax = plt.subplots()

loss_list = []
for ep in range(50) :
    inner_pbar = tqdm(train_dataloader)
    for X, y in inner_pbar:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(X)
        loss = torch.nn.L1Loss()(nn.Sigmoid()(y_pred), y)
        loss.backward()
        optimizer.step()

        inner_pbar.set_postfix({"L1": "{:.6f}".format(loss.item())})
        loss_list.append(loss.item())

ax.plot(range(len(loss_list)), loss_list, color='red', label='MAE')
ax.set_title('Erro de treinamento do modelo')
ax.set_xlabel('Número de Interações')
ax.set_ylabel('MAE')
ax.legend()

plt.savefig('/mnt/ramdisk/treinamento_siren.png')

ax.clear()

predictions = []
true_y = []

inner_pbar = tqdm(val_dataloader)
for X, y in inner_pbar:
    X = X.to(device)
    y = y.to(device)

    y_pred = model(X)
    predictions.append(y_pred)
    true_y.append(y)

    inner_pbar.set_postfix({"L1": "{:.6f}".format(loss.item())})

predictions = numpy .concatenate([p.cpu().detach().numpy() for p in predictions], axis=0)
true_y = numpy.concatenate([y.cpu().detach().numpy() for y in true_y], axis=0)

predictions = scaler.inverse_transform(predictions)
true_y = scaler.inverse_transform(true_y)

predictions = predictions.mean()
true_y = true_y.mean()

L1 = numpy.abs(predictions - true_y)
print(f"L1 = {L1:.4f}")
