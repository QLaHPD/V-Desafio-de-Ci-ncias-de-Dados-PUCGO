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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

#INPUT SHAPES [320 Image, (384, 384, 384) Texts, 3 spatio-temporal, 10 numeric data, 30 categoric data]
warnings.filterwarnings("ignore")


def calc_model_params(model, name):
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{name} parameters count: {num_params}')

def get_image(image_name):
    path = os.path.join('/home/visilionosh/Documents/V DESAFIO CIENCIA DE DADOS/imagens', image_name)
    if os.path.exists(path):
        image = cv2.imread(path)
        h, w, c = image.shape
        if h != 144 or w != 216:
            image = cv2.resize(image, (216, 144), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).to(torch.float32)
    else:
        return torch.zeros(3, 144, 216)

def collate_fn(list_items):
    x , y = [], []
    for x_, y_ in list_items:
        x.append(x_)
        y.append(y_)
    return x, y

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.linear = nn.Linear(input_size, output_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_size,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        if self.input_size != self.output_size:
            x = self.linear(x)
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x)
        return x

class Dense(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout):
        super(Dense, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, output_size))
            else:
                layers.append(nn.Linear(output_size, output_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.image_branch = TransformerEncoder(
            input_size=320,
            output_size=32,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        self.text_branch = TransformerEncoder(
            input_size=384,
            output_size=64,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
        self.numerical_branch = TransformerEncoder(
            input_size=24,
            output_size=16,
            num_heads=4,
            num_layers=3,
            dropout=0.1
        )
        self.one_hot_branch = Dense(
            input_size=65,
            output_size=16,
            num_layers=3,
            dropout=0.1
        )
        self.final_branch = Dense(
            input_size=256,
            output_size=16,
            num_layers=3,
            dropout=0.1
        )

    def forward(self, image_data, text_data_1, text_data_2, text_data_3, numerical_data, one_hot_data):
        image_output = self.image_branch(image_data)
        text_output_1 = self.text_branch(text_data_1)
        text_output_2 = self.text_branch(text_data_2)
        text_output_3 = self.text_branch(text_data_3)
        text_output = torch.cat([text_output_1, text_output_2, text_output_3], dim=1)
        numerical_output = self.numerical_branch(numerical_data)
        one_hot_output = self.one_hot_branch(one_hot_data)

        combined_output = torch.cat([image_output, text_output, numerical_output, one_hot_output], dim=1)
        final_output = self.final_branch(combined_output)
        final_output = torch.mean(final_output, dim=1)


        return final_output



class CustomDataset(Dataset):
    def __init__(self, number_df, text_df, image_df, y):
        self.image_df = image_df
        self.text_df = text_df
        self.number_df = number_df
        self.y = y

    def __len__(self):
        return len(self.number_df)

    def __getitem__(self, idx):
        img_row = self.image_df.iloc[idx]
        picture_name = img_row['picture_name']
        image = get_image(picture_name)

        text_row = self.text_df.iloc[idx].fillna("")
        text = []
        for col in ["description", "house_rules", "amenities"]:
            if text_row[col] == "":
                text.append("basic")
            else:
                text.append(text_row[col])


        number = []
        for col in range(self.number_df.shape[1]):
            number.append(float(self.number_df[idx, col]))

        X = [image, text, number]
        y = float(self.y[idx])
        return X, y




dataframe = pd.read_csv('/mnt/ramdisk/dados_combinados_R1.11.csv').drop(['id', 'last_scraped [norm.]', ' [host_verifications]'], axis=1)
X = dataframe.drop('price', axis=1)
y = dataframe['price']
del dataframe

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

X_text_train, X_text_val = X_train.drop(X_train.columns.difference(["description", "house_rules", "amenities"]), axis=1), X_val.drop(X_val.columns.difference(["description", "house_rules", "amenities"]), axis=1)
X_image_train, X_image_val = X_train.drop(X_train.columns.difference(["picture_name"]), axis=1), X_val.drop(X_val.columns.difference(["picture_name"]), axis=1)
X_train.drop(["description", "house_rules", "amenities", "picture_name"], axis=1, inplace=True)
X_val.drop(["description", "house_rules", "amenities", "picture_name"], axis=1, inplace=True)

ct = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["host_response_time", "host_response_rate",
        "host_total_listings_count", "latitude", "longitude",
        "accommodates", "bathrooms", "bedrooms", "beds", "security_deposit", "cleaning_fee",
        "guests_included", "extra_people", "minimum_nights", "maximum_nights",
        "availability_30", "availability_60", "availability_90",
        "availability_365", "number_of_reviews", "number_of_reviews_ltm",
        "cancellation_policy", "reviews_per_month"]),

        ("pass", "passthrough", ['last_scraped [norm. PI]',
        'host_identity_verified', 'facebook [host_verifications]', 'reviews [host_verifications]',
        'work_email [host_verifications]', 'jumio [host_verifications]',
        'government_id [host_verifications]',
        'manual_offline [host_verifications]',
        'offline_government_id [host_verifications]',
        'selfie [host_verifications]', 'identity_manual [host_verifications]',
        'google [host_verifications]', 'manual_online [host_verifications]',
        'linkedin [host_verifications]', 'kba [host_verifications]',
        'sent_id [host_verifications]', 'photographer [host_verifications]',
        'Entire home/apt [room_type]', 'Private room [room_type]',
        'Shared room [room_type]', 'Hotel room [room_type]',
        'Apartment [property_type]', 'House [property_type]',
        'Serviced apartment [property_type]', 'Condominium [property_type]',
        'Cabin [property_type]', 'Bungalow [property_type]',
        'Aparthotel [property_type]', 'Loft [property_type]',
        'Bed and breakfast [property_type]', 'Hotel [property_type]',
        'Tent [property_type]', 'Villa [property_type]',
        'Guest suite [property_type]', 'Guesthouse [property_type]',
        'Castle [property_type]', 'Cottage [property_type]',
        'Other [property_type]', 'Hostel [property_type]',
        'Casa particular (Cuba) [property_type]',
        'Vacation home [property_type]', 'Townhouse [property_type]',
        'Pousada [property_type]', 'Boat [property_type]',
        'Nature lodge [property_type]', 'Earth house [property_type]',
        'Dorm [property_type]', 'Campsite [property_type]',
        'Casa particular [property_type]', 'Tiny house [property_type]',
        'Chalet [property_type]', 'Barn [property_type]',
        'Boutique hotel [property_type]', 'Resort [property_type]',
        'Hut [property_type]', 'Timeshare [property_type]',
        'Igloo [property_type]', 'Tipi [property_type]',
        'In-law [property_type]', 'Island [property_type]',
        'Treehouse [property_type]', 'Pension (South Korea) [property_type]',
        'Camper/RV [property_type]', 'Farm stay [property_type]',
        'Dome house [property_type]', 'host_is_superhost'])
    ])

X_train = ct.fit_transform(X_train)
X_val = ct.transform(X_val)

scaler = StandardScaler()
scaler.fit(y_train.values.reshape(-1, 1))
y_train, y_val = scaler.transform(y_train.values.reshape(-1, 1)), scaler.transform(y_val.values.reshape(-1, 1))
y_train = nn.Sigmoid()(torch.from_numpy(y_train.squeeze(1)))

train_dataset = CustomDataset(X_train, X_text_train, X_image_train, y_train)
val_dataset = CustomDataset(X_val, X_text_val, X_image_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn)


model = Model()
model.to('cuda')

text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to('cuda')
text_model.max_seq_length = 256

image_model= timm.create_model('convnextv2_atto.fcmae', pretrained=True, num_classes=0).to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

fig, ax = plt.subplots()

loss_list = []
inner_pbar = tqdm(train_dataloader)
for X, y in inner_pbar:
    y = torch.tensor(y).to('cuda')
    images = torch.stack([x[0] for x in X]).to('cuda')
    images = image_model(images)

    texts_list = [[x[1] for x in X]]
    texts_list = texts_list[0]
    texts_tensor = torch.empty((0, 3, 384))
    for i in range(len(texts_list)):
        strs = texts_list[i]
        output = text_model.encode(strs)
        output = numpy.expand_dims(output, 0)
        texts_tensor = torch.cat((texts_tensor, torch.from_numpy(output)), dim=0)
    texts_tensor = texts_tensor.to('cuda')

    number_list = [x[2] for x in X]
    number_tensor = torch.empty((0, 89))
    for i in range(len(number_list)):
        numbers = number_list[i]
        numbers = torch.tensor(numbers, dtype=torch.float)
        number_tensor = torch.cat((number_tensor, numbers.unsqueeze(0)), dim=0)

    number_tensor.to('cuda')
    numerical_tensor = number_tensor[:, :24].to('cuda')
    one_hot_tensor = number_tensor[:, 24:].to('cuda')

    del X, number_tensor

    optimizer.zero_grad()

    y_pred = nn.Sigmoid()(model(images, texts_tensor[:, 0], texts_tensor[:, 1], texts_tensor[:, 2], numerical_tensor, one_hot_tensor))
    loss = torch.nn.L1Loss()(y_pred, y)
    loss.backward()
    optimizer.step()

    inner_pbar.set_postfix({"L1": "{:.6f}".format(loss.item())})
    loss_list.append(loss.item())

print(sum(loss_list) / len(loss_list))

ax.plot(range(len(loss_list)), loss_list, color='red', label='MAE')
ax.set_title('Erro de treinamento do modelo')
ax.set_xlabel('Número de Interações')
ax.set_ylabel('MAE')
ax.legend()

plt.savefig('/mnt/ramdisk/treinamento.png')

ax.clear()


for X, y in val_dataloader:
    y = torch.tensor(y).to('cuda')
    images = torch.stack([x[0] for x in X]).to('cuda')
    images = image_model(images)

    texts_list = [[x[1] for x in X]]
    texts_list = texts_list[0]
    texts_tensor = torch.empty((0, 3, 384))
    for i in range(len(texts_list)):
        strs = texts_list[i]
        output = text_model.encode(strs)
        output = numpy.expand_dims(output, 0)
        texts_tensor = torch.cat((texts_tensor, torch.from_numpy(output)), dim=0)
    texts_tensor = texts_tensor.to('cuda')

    number_list = [x[2] for x in X]
    number_tensor = torch.empty((0, 89))
    for i in range(len(number_list)):
        numbers = number_list[i]
        numbers = torch.tensor(numbers, dtype=torch.float)
        number_tensor = torch.cat((number_tensor, numbers.unsqueeze(0)), dim=0)

    number_tensor.to('cuda')
    numerical_tensor = number_tensor[:, :24].to('cuda')
    one_hot_tensor = number_tensor[:, 24:].to('cuda')

    del X, number_tensor

    y_pred = nn.Sigmoid()(model(images, texts_tensor[:, 0], texts_tensor[:, 1], texts_tensor[:, 2], numerical_tensor, one_hot_tensor))
    loss = torch.nn.L1Loss()(y_pred, y)

    inner_pbar.set_postfix({"L1": "{:.6f}".format(loss.item())})
    loss_list.append(loss.item())

print(sum(loss_list) / len(loss_list))

ax.plot(range(len(loss_list)), loss_list, color='red', label='MAE')
ax.set_title('Erro de treinamento do modelo')
ax.set_xlabel('Número de Interações')
ax.set_ylabel('MAE')
ax.legend()

plt.savefig('/mnt/ramdisk/validação.png')

torch.save(model.state_dict(), 'model.pth')
