import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ------------------------------ Data treatment ------------------------------ #
def get_technical_df():
    technical_df = pd.read_csv("Data/Nasdaq-100 Technical/NASDAQ_100.csv", sep=",")
    technical_df = filter_date_df(technical_df)
    technical_df.drop(columns=["avg_vol_20d", "change_percent", "date"], inplace=True)

    return technical_df

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
    filtered_df = df[(df['date'] >= "2015-01-01") & (df['date'] <= "2022-12-31")]

    return filtered_df


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

# def create_sequences(data, target_index, sequence_length, prediction_horizon):
#     X, y = [], []
    
#     for i in range(len(data) - sequence_length - prediction_horizon):
#         seq_X = data[i:i + sequence_length]  # Fenêtre d'entrée (ex: 2 ans)
#         future_avg = np.mean(data[i + sequence_length : i + sequence_length + prediction_horizon, target_index])
#         current_price = data[i + sequence_length - 1, target_index]
#         seq_y = 1 if future_avg > current_price else 0  # Hausse ou baisse sur 1 an
        
#         X.append(seq_X)
#         y.append(seq_y)

#     return np.array(X), np.array(y)

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