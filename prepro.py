import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Charger les données
x_train = pd.read_csv(r'data/x_train_final.csv')
y_train = pd.read_csv(r'data/y_train_final_j5KGWWK.csv')
x_test = pd.read_csv(r'data/x_test_final.csv')

feature_columns = [
                    # Variables contextuelles
                    'train','gare','date','arret',
                    # Variables passées
                    'p2q0','p3q0','p4q0','p0q2','p0q3','p0q4',
                    # meteo 
                    'precipitation_sum (mm)','temperature_2m_mean (°C)', 'wind_speed_10m_max (km/h)','snowfall_sum (cm)', 'rain_sum (mm)',

                    # mouvements
                    'mouv_social','tx grevistes',

                    # incidents
                    'Gravité EPSF',

                    # avis
                    'Global', 'Clients', 'Non-Clients',

                    # accident travail
                    'Nombre Accidents', "Nombre de jours d'absences", 'Taux de fréquence mensuel','Taux de fréquence annuel glissant 12 mois', 'Taux de gravité mensuel','Taux de gravité annuel glissant 12 mois', 

                    ]

# Suppression de la colonne date originale après extraction des features
x_train = x_train.drop(columns=['Unnamed: 0.1','Unnamed: 0'], axis=1)
y_train = y_train.drop('Unnamed: 0', axis=1)
x_test = x_test.drop('Unnamed: 0', axis=1)

x_meteo = pd.read_csv(r'data/open-meteo.csv')

x_accident_voyageur =  pd.read_csv(r'data/incidents-de-securite.csv', sep = ";")
x_accident_voyageur = x_accident_voyageur.groupby(['Date'])[['Gravité EPSF']].sum().reset_index()

x_greve =  pd.read_csv(r'data/mouvements-sociaux.csv', sep = ";")

x_avis =  pd.read_csv(r'data/avis-data.csv', sep = ";")
x_avis = x_avis[x_avis['Indicateurs']=='Ponctualité']
x_avis = x_avis.groupby('Date')[['Global','Clients','Non-Clients']].mean()

x_accident_travail =  pd.read_csv(r'data/accident-travail-taux.csv', sep = ";")

df_feature_sup = x_meteo.merge(x_greve,how='left',right_on='Date',left_on='time').drop(columns='Date',axis=1)
df_feature_sup = df_feature_sup.merge(x_accident_voyageur,how='left',right_on='Date',left_on='time').drop(columns='Date',axis=1)

df_feature_sup_mensuelle = x_avis.merge(x_accident_travail,how='left',right_on='Date',left_on='Date')
df_feature_sup['Date'] = pd.to_datetime(df_feature_sup['time'], format='%d/%m/%Y').dt.strftime('%Y-%m')
df_feature_sup['time'] = pd.to_datetime(df_feature_sup['time'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
df_feature_final = df_feature_sup.merge(df_feature_sup_mensuelle, how = 'left', on='Date')#.drop(columns='Date')

x_train = x_train.merge(df_feature_final, how='left', right_on='time',left_on='date').reset_index()
x_test = x_test.merge(df_feature_final, how='left', right_on='time',left_on='date').reset_index()

x_train['Motif exprimé'].fillna(0,inplace=True)
x_train['mouv_social'] = x_train['Motif exprimé'].apply(lambda x: 1 if x!=0 else x)
x_train['tx grevistes'] = x_train['Taux de grévistes au sein de la population concernée par le préavis'].fillna(0)
x_train.drop(columns=['Motif exprimé','Organisations syndicales','Taux de grévistes au sein de la population concernée par le préavis'], inplace=True)
x_train['Gravité EPSF'].fillna(0,inplace=True)

x_test['Motif exprimé'].fillna(0,inplace=True)
x_test['mouv_social'] = x_test['Motif exprimé'].apply(lambda x: 1 if x!=0 else x)
x_test['tx grevistes'] = x_test['Taux de grévistes au sein de la population concernée par le préavis'].fillna(0)
x_test.drop(columns=['Motif exprimé','Organisations syndicales','Taux de grévistes au sein de la population concernée par le préavis'], inplace=True)
x_test['Gravité EPSF'].fillna(0,inplace=True)

x_train['date'] = pd.to_datetime(x_train['date'])
x_train['jour_semaine'] = x_train['date'].dt.day_name()


def outliers(df, colonnes, multiplicateur=3):  
    df_result = df.copy()
    resultats_par_colonne = {}

    for colonne in colonnes:
        Q1 = np.percentile(df_result[colonne], 1)  # 10th percentile
        Q3 = np.percentile(df_result[colonne], 99)  # 90th percentile
        IQR = Q3 - Q1
       
        limite_inf = Q1 - multiplicateur * IQR
        limite_sup = Q3 + multiplicateur * IQR

        # Identify outliers
        outliers_inf = df_result[colonne] < limite_inf
        outliers_sup = df_result[colonne] > limite_sup

        # Compute replacement value (median of non-outliers)
        valeurs_normales = df_result.loc[~(outliers_inf | outliers_sup), colonne]
        remplacement = valeurs_normales.mean()  

        nb_inf = outliers_inf.sum()
        nb_sup = outliers_sup.sum()

        # Replace outliers
        df_result.loc[outliers_inf | outliers_sup, colonne] = remplacement

        # Store summary
        resultats_par_colonne[colonne] = f"{nb_inf + nb_sup} valeurs remplacées par la médiane ({remplacement:.2f})"

    return df_result

continuous_features = ['p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
x_train = outliers(x_train,continuous_features)
x_test = outliers(x_test,continuous_features)

cols_categorical = ['train','gare']

label_encoders = {}
for col in cols_categorical:
    le = LabelEncoder()
    x_train[col] = le.fit_transform(x_train[col])  # Transformation
    x_test[col] = le.fit_transform(x_test[col])
    label_encoders[col] = le

# Conversion de la date en features temporelles
def prepare_data(X, columns):
    # Préparation des features
    X = X[columns]

    # Conversion de la date en features temporelles
    X['date'] = pd.to_datetime(X['date'])
    X['jour_semaine'] = X['date'].dt.dayofweek
    X['mois'] = X['date'].dt.month

    # Suppression de la colonne date originale après extraction des features
    X = X.drop('date', axis=1)

    return X

x_train = prepare_data(x_train, feature_columns)
x_test = prepare_data(x_test, feature_columns)

x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
x_test_split,x_val,y_test_split,y_val = train_test_split(x_test_split,y_test_split,test_size=0.5,random_state=42)

# Afficher les dimensions pour vérification
print(f"x_train_split: {x_train_split.shape}, x_test_split: {x_test_split.shape}, x_val: {x_val.shape}")
print(f"y_train_split: {y_train_split.shape}, y_test_split: {y_test_split.shape}, y_val: {y_val.shape}")