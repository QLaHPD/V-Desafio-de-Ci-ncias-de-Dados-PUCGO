import os
import cv2
import timm
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class GetImages(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.folder, file_name)
        image = cv2.imread(file_path)
        h, w, c = image.shape
        if h != 144 or w != 216:
            image = cv2.resize(image, (216, 144), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).to(torch.float32), file_name



files = '/home/visilionosh/Documents/V DESAFIO CIENCIA DE DADOS/imagens'
dataset = GetImages(files)
dataset = DataLoader(dataset, batch_size=48, shuffle=False, num_workers=12)


identity = lambda x: x
model = timm.create_model('convnextv2_atto', pretrained=True).to('cuda')
model.fc = identity

names = []
features = []

for i in tqdm(dataset):
    output, file_names = model(i[0].to('cuda')).mean(dim=[2, 3]), i[1]
    names.extend(file_names)
    features.extend(output.detach().cpu())


dictionary = {'Names': names, 'Features': features}
torch.save(dictionary, '/mnt/ramdisk/dictionary.pth')
