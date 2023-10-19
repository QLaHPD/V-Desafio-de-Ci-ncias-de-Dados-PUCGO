import os
import warnings
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import minmax_scale, StandardScaler


warnings.filterwarnings("ignore")

file = '/mnt/ramdisk/dados_combinados_R1.10.csv'
df = pd.read_csv(file).sort_values(by='price')
drop_columns = ['id', 'amenities', 'last_scraped [norm.]', 'picture_name',
'host_response_time', 'picture_name', 'email [host_verifications]',
'phone [host_verifications]', 'has_availability',
'host_identity_verified', 'guests_included',
'host_response_rate', 'host_is_superhost',
'host_total_listings_count', 'bedrooms',
'security_deposit', 'cleaning_fee', 'extra_people',
'minimum_nights', 'maximum_nights', 'availability_30',
'availability_90', 'availability_365',
'number_of_reviews_ltm', 'cancellation_policy', 'reviews_per_month',
'facebook [host_verifications]', 'reviews [host_verifications]',
'work_email [host_verifications]', 'jumio [host_verifications]',
'government_id [host_verifications]',
'manual_offline [host_verifications]',
'offline_government_id [host_verifications]',
'selfie [host_verifications]', 'identity_manual [host_verifications]',
'google [host_verifications]', 'manual_online [host_verifications]',
' [host_verifications]', 'linkedin [host_verifications]',
'kba [host_verifications]', 'sesame [host_verifications]',
'sesame_offline [host_verifications]', 'sent_id [host_verifications]',
'photographer [host_verifications]', 'amex [host_verifications]',
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
'Dome house [property_type]', 'Plane [property_type]']
df.drop(columns=drop_columns, axis=1, inplace=True)
#df['security_deposit'] = df['security_deposit'].fillna(0)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
y_train, y_val = y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1)

ct = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ['beds', 'bathrooms', 'availability_60',
        'number_of_reviews', 'accommodates','latitude', 'longitude'
        ]),
        ("pass", "passthrough", ['last_scraped [norm. PI]'])
    ])

X_train = ct.fit_transform(X_train)
X_val = ct.transform(X_val)

scaler = StandardScaler()
scaler.fit(y_train)
y_train, y_val = scaler.transform(y_train), scaler.transform(y_val)

rf = RF(n_estimators=100, max_depth=None, verbose=2, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_val, y_pred = y_val.reshape(-1, 1), y_pred.reshape(-1, 1)
y_val, y_pred = scaler.inverse_transform(y_val), scaler.inverse_transform(y_pred)
print(f'\n{mean_absolute_error(y_val, y_pred)}\t{r2_score(y_val, y_pred)}')

result = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=1, n_jobs=1)
feature_names = df.columns.difference(['price'])
sorted_idx = result.importances_mean.argsort()
fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=[feature_names[i] for i in sorted_idx])
ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
ax.set_title('Variaveis importantes (conjunto de validação)')
fig.tight_layout()
plt.savefig('/mnt/ramdisk/Variaveis_relevantes.png')
