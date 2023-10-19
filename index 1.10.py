import os
import math
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


def modified_z_score(x, threshold=5):
  median_x = np.median(x)
  mad_x = median_abs_deviation(x)
  z_score = 0.6744897502 * (x - median_x) / mad_x
  return np.abs(z_score) > threshold

def normalizar(data):
    min = data.min()
    max = data.max()
    data = (data - min) / (max - min)
    return data

def normalizar_pi(col):
    ncol = []
    for val in col:
        val = (math.sin(((val+2)/12)*math.pi*2)+1)/2
        ncol.append(val)
    return ncol

warnings.filterwarnings("ignore")
diretorio = '/home/visilionosh/Documents/V DESAFIO CIENCIA DE DADOS/Data'
version = '1.11'

dataframes = []

colunas_a_serem_ignoradas = [
    'scrape_id', 'source', 'host_url', 'thumbnail_url', 'medium_url',
    'xl_picture_url', 'host_thumbnail_url', 'host_listings_count',
    'host_has_profile_pic', 'street', 'neighbourhood_cleansed',
    'neighbourhood_group_cleansed', 'market', 'smart_location',
    'country_code', 'country', 'calendar_last_scraped', 'first_review',
    'last_review', 'license', 'jurisdiction_names', 'calculated_host_listings_count_shared_rooms',
    'calendar_updated', 'zipcode', 'city', 'state', 'bed_type', 'square_feet',
    'require_guest_profile_picture', 'require_guest_phone_verification', 'minimum_minimum_nights',
    'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm',
    'maximum_nights_avg_ntm', 'is_business_travel_ready', 'requires_license', 'experiences_offered',
    'calendar_updated', 'host_acceptance_rate', 'calculated_host_listings_count',
    'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_entire_homes',
    'listing_url', 'review_scores_rating', 'review_scores_checkin', 'review_scores_accuracy',
    'review_scores_location', 'review_scores_location', 'review_scores_cleanliness',
    'review_scores_communication', 'review_scores_value', 'host_about', 'host_location',
    'host_neighbourhood', 'neighbourhood', 'host_name', 'name',
    'summary', 'space', 'neighborhood_overview', 'notes', 'transit', 'access',
    'interaction', 'picture_url', 'host_picture_url', 'host_since',
    'is_location_exact', 'host_id', 'weekly_price', 'monthly_price'
    ]

for arquivo in os.listdir(diretorio):
    if arquivo.endswith(".csv"):
        print("Lendo " + arquivo)
        caminho_arquivo = os.path.join(diretorio, arquivo)
        df = pd.read_csv(caminho_arquivo)
        df = df.drop(columns=colunas_a_serem_ignoradas, errors='ignore')

        dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)
del dataframes
df = df.dropna(subset=['description'])
print("DF Bruto:", df.shape[0], "linhas.")

linhas_antes = df.shape[0]
df = df.drop_duplicates()
linhas_depois = df.shape[0]
print(f"Linhas removidas: {linhas_antes - linhas_depois}")

colunas_para_substituir = ['host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']
mapeamento = {'f': 0, 't': 1}
df[colunas_para_substituir] = df[colunas_para_substituir].replace(mapeamento)

df = df.dropna(subset=['price'])
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df = df[df['price'] != 0.0]
x = df['price'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]
df.sort_values(by='price', inplace=True)


df = df.dropna(subset=['last_scraped'])
df['last_scraped'] = pd.to_datetime(df['last_scraped'], format='%Y-%m-%d').dt.strftime('%m/%Y')
df['last_scraped'] = pd.to_datetime(df['last_scraped'], format='%m/%Y')
df['last_scraped [norm. PI]'] = df['last_scraped'].dt.month
df['last_scraped [norm. PI]'] = normalizar_pi(df['last_scraped [norm. PI]'])
df['last_scraped [norm.]'] = normalizar(df['last_scraped']).round(7)
df = df.drop(['last_scraped'], axis=1)

mapeamento = {'within an hour': 0, 'within a few hours': 1/3, 'within a day': (1/3)*2, 'a few days or more': 1}
df['host_response_time'] = df['host_response_time'].replace(mapeamento)
df['host_response_time'] = df['host_response_time'].fillna(-1)

df['host_is_superhost'] = df['host_is_superhost'].fillna(0)

df['host_identity_verified'] = df['host_identity_verified'].fillna(0)

df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float) / 100
df['host_response_rate'] = df['host_response_rate'].fillna(0)
df['host_total_listings_count'] = df['host_total_listings_count'].fillna(1)

df = df.dropna(subset=['latitude'])
df = df.dropna(subset=['longitude'])

x = df['accommodates'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]
media = df['accommodates'].mean()
df['accommodates'] = df['accommodates'].fillna(4)

x = df['bathrooms'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]
media = df['bathrooms'].mean()
df['bathrooms'] = df['bathrooms'].fillna(media)

x = df['bedrooms'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]
media = df['bedrooms'].mean()
df['bedrooms'] = df['bedrooms'].fillna(media)

x = df['beds'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]
media = df['beds'].mean()
df['beds'] = df['beds'].fillna(media)

df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)
df['cleaning_fee'] = df['cleaning_fee'].fillna(0)
x = df['cleaning_fee'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['guests_included'] = df['guests_included'].fillna(1)
x = df['guests_included'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)
df['security_deposit'] = df['security_deposit'].fillna(0)
x = df['security_deposit'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['extra_people'] = df['extra_people'].fillna(0)
df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',', '').astype(float)
x = df['extra_people'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['minimum_nights'] = df['minimum_nights'].fillna(1)
x = df['minimum_nights'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['maximum_nights'] = df['maximum_nights'].fillna(1250)
df.loc[df['maximum_nights'] > 1125, 'maximum_nights'] = 1125

df = df.dropna(subset=['availability_30'])
x = df['availability_30'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df = df.dropna(subset=['availability_60'])
x = df['availability_60'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df = df.dropna(subset=['availability_90'])
x = df['availability_90'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df = df.dropna(subset=['availability_365'])
x = df['availability_365'].to_numpy().astype(float)
outlier = modified_z_score(x)
df = df[~outlier]

df['number_of_reviews'] = df['number_of_reviews'].fillna(0)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['number_of_reviews_ltm'] = df['number_of_reviews_ltm'].fillna(0)

mapeamento = {'flexible': 0, 'moderate': 0.2, 'strict': 0.4, 'strict_14_with_grace_period': 0.6, 'super_strict_30': 0.8, 'super_strict_60': 1}
df['cancellation_policy'] = df['cancellation_policy'].replace(mapeamento)
df = df.dropna(subset=['cancellation_policy'])

df['amenities'] = df['amenities'].str.replace('"', '').str.replace('{', '').str.replace('}', '')


df = df.dropna(subset=['host_verifications'])
df['host_verifications'] = df['host_verifications'].str.replace("'", "").str.replace('[', '').str.replace(']', '').str.replace(' ', '')
df['host_verifications'] = df['host_verifications'].str.split(',')
vars = []
for l in df['host_verifications']:
    for v in l:
        v = v.strip()
        if v not in vars:
            vars.append(v)
for col in vars:
    col_name = col + ' [host_verifications]'
    df[col_name] = df['host_verifications'].apply(lambda x: 1 if col in x else 0)
df = df.drop(columns=['host_verifications'])

df = df.dropna(subset=['room_type'])
vars = []
for v in df['room_type']:
    if v not in vars:
        vars.append(v)
for col in vars:
    col_name = col + ' [room_type]'
    df[col_name] = df['room_type'].apply(lambda x: 1 if col in x else 0)
df = df.drop(columns=['room_type'])

df = df.dropna(subset=['property_type'])
vars = []
for v in df['property_type']:
    if v not in vars:
        vars.append(v)
for col in vars:
    col_name = col + ' [property_type]'
    df[col_name] = df['property_type'].apply(lambda x: 1 if col in x else 0)
df = df.drop(columns=['property_type'])


df['picture_name'] = df['id'].astype(str) + '_' + df['last_scraped [norm.]'].astype(str) + '.jpg'
df_original = pd.read_csv("/home/visilionosh/Documents/V DESAFIO CIENCIA DE DADOS/imagens_coletadas.csv")
verificacao_values = df_original["picture_name"].tolist()
mask = df["picture_name"].isin(verificacao_values)
df.loc[~mask, "picture_name"] = "VAZIO"
del df_original

print("\n")
print("DF Tratado:", df.shape[0], "linhas.")
print(df.columns)
arquivo_saida = f"/mnt/ramdisk/dados_combinados_R{version}.csv"

df.to_csv(arquivo_saida, index=False)
print("\n")
print(f"Arquivo CSV gerado com sucesso: {arquivo_saida}.")

arquivo_saida = f"/mnt/ramdisk/dados_combinados_R{version}_amostra.csv"
indices_aleatorios = random.sample(range(len(df)), 250)
linhas_aleatorias = df.iloc[indices_aleatorios]
df_aleatorio = pd.DataFrame(linhas_aleatorias).sort_values(by='price')

df_aleatorio.to_csv(arquivo_saida, index=False)
print("\n")
print(f"/mnt/ramdisk/Amostra aleatória em CSV gerada com sucesso: {arquivo_saida}.")
