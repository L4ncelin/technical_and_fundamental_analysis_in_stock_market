import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ------------------------------ Data treatment ------------------------------ #
def get_technical_df():
    technical_df = pd.read_csv("Data/Nasdaq-100 Technical/NASDAQ_100_companies.csv", sep="\t")
    #technical_df = filter_date_df(technical_df)
    #technical_df.drop(columns=["avg_vol_20d", "change_percent"], inplace=True)

    return technical_df

def get_fundamental_df():
    fundamental_df = pd.read_csv("Data/Nasdaq-100 Fundamental/nasdaq100_metrics_ratios.csv", sep=",")

    fundamental_df.drop(columns=["company", "sector", "subsector", "predictability", "profitability"], inplace=True)

    return fundamental_df

def compute_indicators(technical_df):

    technical_df["SMA_20"] = technical_df["close"].rolling(window=20).mean()
    technical_df["EMA_20"] = technical_df["close"].ewm(span=20, adjust=False).mean()

    technical_df["Bollinger_Mid"] = technical_df["close"].rolling(window=20).mean()
    technical_df["Bollinger_Std"] = technical_df["close"].rolling(window=20).std()
    technical_df["Bollinger_Upper"] = technical_df["Bollinger_Mid"] + (2 *technical_df["Bollinger_Std"])
    technical_df["Bollinger_Lower"] = technical_df["Bollinger_Mid"] - (2 *technical_df["Bollinger_Std"])

    window_length = 14

    # Calcul des variations de prix
    delta =technical_df["close"].diff()

    # Séparation des gains et des pertes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Moyenne mobile exponentielle des gains et pertes
    avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=1).mean()

    # Calcul du RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calcul du RSI
    technical_df["RSI"] = 100 - (100 / (1 + rs))

    technical_df = technical_df.iloc[19:,:] # Filter 19 because these lines serves to compute indicators with windows
    technical_df.reset_index(drop=True, inplace=True)

    return technical_df


def filter_date_df(df):

    df['date'] = pd.to_datetime(df['date'])

    # We keep only from 2017 to 2022 because of the Fundamental dataset
    filtered_df = df[(df['date'] >= "2017-01-01") & (df['date'] <= "2022-12-31")]

    return filtered_df


def add_technical_indicators(df):
    # Create technical indicators
    df['ATR'] = df.ta.atr(length=20)
    df['RSI'] = df.ta.rsi()
    df['Average'] = df.ta.midprice(length=1) #midprice
    df['MA40'] = df.ta.sma(length=40)
    df['MA80'] = df.ta.sma(length=80)
    df['MA160'] = df.ta.sma(length=160)

    df = df.dropna()

    return df


def scale_df(df, non_scale_var):
    # Save variable names
    var_names = df.columns.tolist()
    for var in non_scale_var:
        var_names.remove(var)

    # Scaling
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.drop(columns=non_scale_var))

    df_scaled = pd.DataFrame(df_scaled, columns=var_names)

    # Reconstruct df with scaled values
    df_new = pd.concat([df[non_scale_var].reset_index(drop=True), df_scaled], axis=1)

    return df_new


def create_windows_dataframe(df, technical_indicators):
    """
    Pour chaque action (colonne 'Name') et pour chaque année entre 2017 et 2021, 
    extrait et sauvegarde dans un DataFrame :
      - 'Name' : le nom de l'action
      - 'year' : l'année traitée (pour la fenêtre technique)
      - 'tech_window' : une liste contenant les 30 derniers jours de l'année en cours (colonne 'Close')
      - 'target_window' : une liste contenant les 3 premiers jours de l'année suivante (colonne 'Close')
      
    Seules les années pour lesquelles il existe au moins 30 jours dans l'année courante
    et au moins 3 jours dans l'année suivante sont retenues.
    
    Paramètres :
      df : DataFrame contenant au moins les colonnes 'date', 'Close', 'Name'
    
    Renvoie :
      windows_df : DataFrame avec les colonnes ['Name', 'year', 'tech_window', 'target_window']
    """
    # S'assurer que la colonne 'date' est bien au format datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Trier par nom d'action et par date
    df = df.sort_values(by=['Name', 'Date']).reset_index(drop=True)
    
    windows = []  # liste de dictionnaires pour chaque fenêtre
    
    # Parcourir chaque action
    for name, group in df.groupby('Name'):
        group = group.copy()
        group['year'] = group['Date'].dt.year
        
        # Pour chaque année de 2017 à 2021
        for yr in sorted(group['year'].unique()):
          if yr < 2016:
            continue
          else:
            data_year = group[group['year'] == yr]
            # Vérifier qu'il y a au moins 30 observations dans l'année
            if len(data_year) < 30:
                continue
            # Fenêtre technique : les 30 derniers jours de l'année (en ordre chronologique)
            tech_window = data_year.iloc[-30:][technical_indicators].to_numpy()
            
            # Récupérer les données de l'année suivante pour constituer la cible
            next_year = yr + 1
            data_next_year = group[group['year'] == next_year]
            if len(data_next_year) < 3:
                continue
            # Fenêtre cible : les 3 premiers jours de l'année suivante
            target_window = data_next_year.iloc[:3]['Close'].tolist()
            
            # Ajouter la fenêtre à la liste
            windows.append({
                'Name': name,
                'year': yr+1,
                'tech_window': tech_window,
                'target_window': target_window
            })
    
    # Conversion de la liste en DataFrame
    windows_df = pd.DataFrame(windows)
    return windows_df


def filter_actions_complete_years(df):
    """
    Filtre le DataFrame pour ne conserver que les actions qui possèdent des fenêtres pour 
    toutes les années de 2017 à 2021.
    
    Paramètres :
      df : DataFrame contenant au moins les colonnes 'Name' et 'year'
    
    Renvoie :
      filtered_df : DataFrame filtré ne contenant que les actions ayant les années 2017 à 2021 complètes.
    """
    required_years = set(range(2017, 2022))  # {2017, 2018, 2019, 2020, 2021}
    valid_actions = []
    
    # Pour chaque action, vérifier si toutes les années requises sont présentes
    for name, group in df.groupby("Name"):
        years_present = set(group["year"].unique())
        if required_years.issubset(years_present):
            valid_actions.append(name)
    
    # Filtrer le DataFrame pour ne garder que les actions valides
    filtered_df = df[df["Name"].isin(valid_actions)].reset_index(drop=True)
    return filtered_df


def transform_fundamentals(df):
    """
    Transforme le DataFrame de fondamentaux de format large en un format long.
    
    Le DataFrame d'entrée doit contenir :
      - Une colonne 'symbol' avec le nom de l'action.
      - Des colonnes pour chaque indicateur avec un suffixe d'année, par exemple :
          asset_turnover_2017, asset_turnover_2018, ..., asset_turnover_2022, asset_turnover_latest.
    
    La fonction effectue les opérations suivantes :
      1. Exclut les colonnes se terminant par '_latest'.
      2. Passe du format large au format long via un melt.
      3. Extrait, à partir du nom de la colonne, le nom de l'indicateur et l'année.
      4. Effectue un pivot pour obtenir une ligne par couple (symbol, year) avec une colonne par indicateur.
    
    Paramètres :
      df (DataFrame) : DataFrame initial contenant la colonne 'symbol' et les indicateurs.
      
    Retourne :
      df_transformed (DataFrame) : DataFrame transformé avec les colonnes 'symbol', 'year' et une colonne par indicateur.
    """
    # On garde la colonne 'symbol' et toutes les colonnes qui ne se terminent pas par '_latest'
    id_vars = ['symbol']
    value_vars = [col for col in df.columns if col not in id_vars and not col.endswith('_latest')]
    
    # Transformation en format long
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, 
                      var_name='variable', value_name='value')
    
    # Extraire l'année et le nom de l'indicateur à partir du nom de la variable.
    # On suppose que les colonnes sont de la forme "indicator_year"
    df_long['year'] = df_long['variable'].apply(lambda x: x.split('_')[-1])
    df_long['indicator'] = df_long['variable'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    # Pivot pour obtenir une ligne par (symbol, year) et une colonne par indicateur
    df_pivot = df_long.pivot_table(index=['symbol', 'year'], 
                                   columns='indicator', 
                                   values='value', 
                                   aggfunc='first').reset_index()
    
    # Convertir l'année en entier
    df_pivot['year'] = df_pivot['year'].astype(int)
    
    # Réordonner les colonnes pour avoir 'symbol' et 'year' en premier
    cols = df_pivot.columns.tolist()
    cols = ['symbol', 'year'] + [col for col in cols if col not in ['symbol', 'year']]
    df_transformed = df_pivot[cols]
    
    return df_transformed


def fill_missing_values(df):
    """
    Remplit les valeurs manquantes dans le DataFrame en interpolant linéairement pour chaque action.
    S'il reste des NaN, applique un forward fill (ffill) puis un backward fill (bfill).
    
    Paramètres :
        df (pd.DataFrame) : DataFrame contenant les colonnes 'Name', 'year' et les indicateurs financiers.

    Retourne :
        pd.DataFrame : DataFrame sans NaN.
    """
    df_filled = df.copy()

    # Appliquer l'interpolation et les fills sur chaque action individuellement
    df_filled = df_filled.groupby("Name").apply(lambda group: group.interpolate(method='linear').ffill().bfill())

    # Réinitialiser l'index pour éviter une multi-indexation
    df_filled.reset_index(drop=True, inplace=True)

    return df_filled


def fill_nan_with_yearly_mean(df):
    """
    Remplace les colonnes entièrement remplies de NaN pour une action donnée par la moyenne de cette variable 
    sur l'année correspondante.

    Paramètres :
        df (pd.DataFrame) : DataFrame contenant les colonnes 'Name', 'year' et les indicateurs financiers.

    Retourne :
        pd.DataFrame : DataFrame où les valeurs entièrement NaN pour une action sont remplacées par la moyenne de l'année correspondante.
    """
    # Identifier les colonnes indicateurs (hors 'Name' et 'year')
    cols_to_fill = df.columns.difference(['Name', 'year'])

    # Calculer la moyenne de chaque variable pour chaque année
    yearly_means = df.groupby("year")[cols_to_fill].mean()

    def fill_action_nan(group):
        """
        Pour chaque action, remplace les colonnes entièrement NaN par la moyenne annuelle correspondante.
        """
        for col in cols_to_fill:
            if group[col].isna().all():  # Vérifie si la colonne est entièrement NaN pour cette action
                group[col] = group['year'].map(yearly_means[col])  # Remplace par la moyenne de l'année correspondante
        return group

    # Appliquer le remplissage à chaque action
    df_filled = df.groupby("Name", group_keys=False).apply(fill_action_nan)

    return df_filled


def prepare_model_inputs(df, tech_features, fund_features):
    """
    Prépare les entrées et sorties pour le modèle.

    :param df: DataFrame contenant les colonnes 'tech_window', 'target_window' et les features fondamentaux
    :param tech_features: Nombre de caractéristiques techniques utilisées dans `tech_window`
    :param fund_features: Liste des noms des variables fondamentales utilisées
    :return: (X_tech, X_fund, y)
    """
    # Convertir tech_window et target_window en numpy array
    X_tech = np.stack(df['tech_window'].values)  # (nb_samples, 30)
    y = np.stack(df['target_window'].values)     # (nb_samples, 3)

    # Vérifier si chaque séquence de `tech_window` a bien la forme attendue
    assert X_tech.shape[1] == 30, f"Erreur : X_tech doit avoir 30 timesteps, mais a {X_tech.shape[1]}"
    
    # Reshape en (nb_samples, timesteps, n_tech_features) avec n_tech_features = 1 si pas d'autres features
    X_tech = X_tech.reshape(X_tech.shape[0], 30, tech_features)

    # Extraire les variables fondamentales
    X_fund = df[fund_features].values  # (nb_samples, n_fund_features)

    # Reshape la sortie `y` pour qu'elle soit en (nb_samples, 3, 1)
    y = y.reshape(y.shape[0], 3, 1)

    return X_tech, X_fund, y


def split_data_by_year(df, train_years, val_years, test_years):
    """
    Sépare les données en train, validation et test en fonction des années.

    :param df: DataFrame avec une colonne 'year'
    :param train_years: Liste des années pour l'entraînement
    :param val_years: Liste des années pour la validation
    :param test_years: Liste des années pour le test
    :return: (df_train, df_val, df_test)
    """
    df_train = df[df['year'].isin(train_years)]
    df_val = df[df['year'].isin(val_years)]
    df_test = df[df['year'].isin(test_years)]
    
    return df_train, df_val, df_test


# ------------------------------ Model computing ----------------------------- #

def get_scaled_split(df):
    # --------------------------------- Découpage -------------------------------- #
    # Découper en train (70%) et reste (30%)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=False)

    # Découper le reste en validation (10%) et test (20%)
    val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, shuffle=False)  # 20% de df

    # Vérification des tailles
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # ------------------------------- Normalization ------------------------------ #
    scaler = MinMaxScaler(feature_range=(0,1))

    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    val_df = scaler.transform(val_df)

    return train_df, test_df, val_df, scaler

def create_sequences(data, target_index, sequence_length, prediction_steps):

    X = []
    y = []

    # Parcours des données pour créer des séquences
    for i in range(len(data) - sequence_length - prediction_steps):
        # Séquence d'entrée (features)
        seq_X = np.delete(data[i:i + sequence_length], target_index, axis=1)
        # Séquence cible (target uniquement)
        seq_y = data[i + sequence_length:i + sequence_length + prediction_steps, target_index]
        X.append(seq_X)
        y.append(seq_y)

    return np.array(X), np.array(y)

def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    """
    Calcule l'accuracy pour un modèle de régression, en fonction d'un seuil de tolérance.

    :param y_true: Valeurs réelles (dénormalisées si nécessaire)
    :param y_pred: Prédictions du modèle (dénormalisées si nécessaire)
    :param tolerance: Seuil d'erreur tolérée pour considérer une prédiction comme correcte (par défaut 5%)
    :return: L'accuracy en pourcentage
    """
    # Calcul de l'erreur absolue entre les prédictions et les vraies valeurs
    error = np.abs(y_true - y_pred)

    # Compter le nombre de prédictions correctes (erreur sous le seuil)
    correct_predictions = np.sum(error < tolerance)

    # Calcul de l'accuracy
    accuracy = correct_predictions / len(y_true) * 100

    return accuracy

def smape(y_true, y_pred):
    # Éviter la division par zéro en ajustant les petites valeurs
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred) / (denominator + np.finfo(float).eps)  # Ajout de np.finfo(float).eps pour éviter la division par zéro
    return 200 * np.mean(diff)

def get_performance_metrics(y_test, y_pred):
    # Calcul des métriques
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    nmae = mae / np.mean(np.abs(y_test))
    accuracy = calculate_accuracy(y_test, y_pred, tolerance=0.05)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    smape_value = smape(y_test, y_pred)

    # Affichage des résultats dans une seule fenêtre
    print(f'----- Résultats des métriques -----')
    print(f'RMSE : {rmse}')
    print(f'MAE : {mae}')
    print(f'R² : {r2}')
    print(f'NMAE : {nmae}')
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE : {smape_value:.2f}%")

def get_performance_metrics_multiple_days(y_test, y_pred):
    """
    Calcule les métriques de performance pour chaque jour de prédiction.

    :param y_test: Vérités terrain (shape: batch_size, 3, 1)
    :param y_pred: Prédictions du modèle (shape: batch_size, 3)
    :return: Dictionnaire contenant les métriques pour chaque horizon de prédiction
    """
    # Reshape y_test pour qu'il ait la même forme que y_pred
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    metrics = {}
    for day in range(y_test.shape[1]):  # Pour chaque jour de prédiction
        metrics[f"Day {day+1}"] = {
            "RMSE": np.sqrt(mean_squared_error(y_test[:, day], y_pred[:, day])),
            "MAE": mean_absolute_error(y_test[:, day], y_pred[:, day]),
            "R²": r2_score(y_test[:, day], y_pred[:, day]),
            "NMAE": mean_absolute_error(y_test[:, day], y_pred[:, day]) / np.mean(np.abs(y_test[:, day])),
            "Accuracy": calculate_accuracy(y_test[:, day], y_pred[:, day], tolerance=0.05),
            "MAPE": mean_absolute_percentage_error(y_test[:, day], y_pred[:, day]) * 100,
            "SMAPE": smape(y_test[:, day], y_pred[:, day])
        }

    return metrics

# ------------------------------- Graph results ------------------------------ #

def plot_score_per_day(y_test, y_pred):
    # Initialize lists to store metrics
    r2_scores = []
    mae_scores = []
    accuracy_percentages = []

    # Loop through each forecast day
    for day in range(y_test.shape[1]):  # Iterate over forecast horizon
        r2 = r2_score(y_test[:, day], y_pred[:, day])
        mae = mean_absolute_error(y_test[:, day], y_pred[:, day])
        mean_actual = np.mean(y_test[:, day])
        accuracy = 100 - (mae / mean_actual * 100)

        r2_scores.append(r2)
        mae_scores.append(mae)
        accuracy_percentages.append(accuracy)

    # Plot R² Scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, y_test.shape[1] + 1), r2_scores, marker='o', label="R² Score")
    plt.title("R² Score for Each Forecast Day", fontsize=16)
    plt.xlabel("Days Ahead in Forecast", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot MAE Scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, y_test.shape[1] + 1), mae_scores, marker='o', color='orange', label="Mean Absolute Error")
    plt.title("Mean Absolute Error (MAE) for Each Forecast Day", fontsize=16)
    plt.xlabel("Days Ahead in Forecast", fontsize=12)
    plt.ylabel("Mean Absolute Error (Normalized)", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Plot Accuracy Percentages
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, y_test.shape[1] + 1), accuracy_percentages, marker='o', color='green', label="Accuracy (%)")
    plt.title("Accuracy Percentage for Each Forecast Day", fontsize=16)
    plt.xlabel("Days Ahead in Forecast", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # Print accuracy percentages for reference
    for day, acc in enumerate(accuracy_percentages, start=1):
        print(f"Day {day}: Accuracy = {acc:.2f}%")

def plot_predictions(y_test, y_pred, window_size=100):
    """
    Affiche les vraies valeurs et les prédictions des prix d'actions en respectant la temporalité.
    
    Paramètres :
    - y_test : np.array de forme (N, 3), valeurs réelles
    - y_pred : np.array de forme (N, 3), valeurs prédites
    - window_size : int, nombre d'échantillons à afficher
    """
    plt.figure(figsize=(14, 8))

    # Vérification pour éviter d'afficher plus que la taille des données
    max_samples = min(len(y_test), window_size)

    # Tracer les vraies valeurs pour chaque jour de prévision
    plt.plot(range(max_samples), y_test[:max_samples, 0], label="Y true (Jour +1)", alpha=0.7, linewidth=1.5)
    #plt.plot(range(max_samples), y_test[:max_samples, 1], label="Y true (Jour +2)", alpha=0.7, linewidth=1.5)
    #plt.plot(range(max_samples), y_test[:max_samples, 2], label="Y true (Jour +3)", alpha=0.7, linewidth=1.5)

    # Tracer les prédictions en respectant le décalage temporel
    plt.plot(range(max_samples), y_pred[:max_samples, 0], '--', label="Y predicted (Jour +1)", alpha=0.7, linewidth=1.5)
    #plt.plot(range(1, max_samples), y_pred[:max_samples-1, 1], '--', label="Y predicted (Jour +2)", alpha=0.7, linewidth=1.5)
    #plt.plot(range(max_samples), y_pred[:max_samples, 2], '--', label="Y predicted (Jour +3)", alpha=0.7, linewidth=1.5)

    # Ajout des détails
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title("Actual vs Predicted Stock Prices", fontsize=16, fontweight='bold')
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Stock Price", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    # Affichage
    plt.show()

def plot_training_curves(history, type_val:str="mse"):
    # Exemple de récupération des valeurs du mse du train et du val (validation)
    train = history.history[type_val]
    val = history.history['val_'+type_val]

    # Créer un graphique
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='Train '+type_val, color='blue')
    plt.plot(val, label='Validation '+type_val, color='orange')

    # Ajouter des labels et un titre
    plt.title('Évolution de la mse du modèle')
    plt.xlabel('Epochs')
    plt.ylabel(type_val)
    plt.legend()

    # Afficher le graphique
    plt.grid(True)
    plt.show()

def plot_errors_hist(y_test, y_pred):
    # Calculer les erreurs
    errors = y_test - y_pred  # y_test : valeurs réelles, y_pred : prédictions

    plt.figure(figsize=(10,6))

    # Histogramme des erreurs
    plt.hist(errors, bins=20, color='blue', edgecolor="black", alpha=0.7, label='Erreurs')

    plt.title('Comparaison des histogrammes des erreurs et des valeurs')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.show()

def plot_errors_hist_multiple_days(y_test, y_pred):
    """
    Affiche les histogrammes des erreurs pour chaque jour de prédiction (Day 1, Day 2, Day 3).

    :param y_test: Vérités terrain (shape: batch_size, 3, 1)
    :param y_pred: Prédictions du modèle (shape: batch_size, 3)
    """
    # Reshape y_test pour qu'il ait la même forme que y_pred
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

    # Calcul des erreurs (différences entre y_test et y_pred)
    errors = y_test - y_pred  # y_test : valeurs réelles, y_pred : prédictions

    # Configuration de la figure pour afficher 3 histogrammes
    plt.figure(figsize=(15, 5))

    # Pour chaque jour de prédiction, afficher l'histogramme des erreurs
    for day in range(y_test.shape[1]):
        plt.subplot(1, 3, day + 1)  # Crée une grille 1x3 pour 3 histogrammes
        plt.hist(errors[:, day], bins=20, color='blue', edgecolor="black", alpha=0.7, label=f'Erreurs Day {day+1}')
        plt.title(f'Histogramme des erreurs - Day {day+1}')
        plt.xlabel('Erreur')
        plt.ylabel('Fréquence')
        plt.legend()

    # Affichage des histogrammes
    plt.tight_layout()
    plt.show()