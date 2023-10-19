import os
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


file = '/mnt/ramdisk/dados_combinados_R1.9.csv'
df = pd.read_csv(file)
df.drop(df.columns.difference(['id', 'description', 'house_rules']), axis=1, inplace=True)
df = df.dropna(subset=['description']).sort_values(by='id')


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to('cuda')
model.max_seq_length = 512

Null_embedding = model.encode('Basic')

dictionary = {}
num_rows = len(df)
for index, row in tqdm(df.iterrows(), total = num_rows):
    if row['house_rules'] == '':
        dictionary[row['id']] = (model.encode(row['description']), Null_embedding)
    else:
        embeddings = model.encode([row['description'], row['house_rules']])
        dictionary[row['id']] = (embeddings[0], embeddings[1])

torch.save(dictionary, '/mnt/ramdisk/text_embeddings.pth')
