import os
import warnings
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import minmax_scale, StandardScaler


def plot_results(y_val, y_pred, i, name, subject):
    y_val, y_pred = y_val.reshape(-1, 1), y_pred.reshape(-1, 1)
    y_val, y_pred = scaler.inverse_transform(y_val), scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mae_list.append(mae)
    r2_list.append(r2)
    i_list.append(i)

    df = pd.DataFrame({'i': i_list, 'MAE': mae_list, 'R2': r2_list})

    fig, ax1 = plt.subplots()
    plt.title('Performace da Random Forest variando ' + subject)
    sns.lineplot(data=df, x='i', y='MAE', label='MAE', color='blue', ax=ax1)
    ax1.set_xlabel(name)
    ax1.set_ylabel('MAE')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='i', y='R2', label='R^2', color='red', ax=ax2)
    ax2.set_ylabel
    plt.savefig(os.path.join('/mnt/ramdisk', name + '.png'))


warnings.filterwarnings("ignore")

file = '/mnt/ramdisk/dados_combinados_R1.10.csv'
df = pd.read_csv(file).sort_values(by='price')
id = df['id']
df.drop(df.columns.difference(['price', 'last_scraped [norm. PI]',
 'latitude', 'longitude']), axis=1, inplace=True)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
y_train, y_val = y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1)

ct = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["latitude", "longitude"]),
        ("pass", "passthrough", ["last_scraped [norm. PI]"])
    ])

X_train = ct.fit_transform(X_train)
X_val = ct.transform(X_val)

scaler = StandardScaler()
scaler.fit(y_train)
y_train, y_val = scaler.transform(y_train), scaler.transform(y_val)

mae_list, r2_list, i_list = [], [], []
for i in range(1, 101):
    rf = RF(n_estimators=i, max_depth=50, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    plot_results(y_val, y_pred, i, 'Estimadores', 'o numero de estimadores')

mae_list, r2_list, i_list = [], [], []
for i in range(1, 51):
    rf = RF(n_estimators=100, max_depth=i, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    plot_results(y_val, y_pred, i, 'Profundidade', 'a profundidade maxima')
