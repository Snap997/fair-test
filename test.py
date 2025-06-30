import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# --- Funzione per calcolare statistiche locali: joint e marginali ---
def compute_local_stats(y, A):
    """
    Calcola le distribuzioni congiunta p(A, Y) e marginale p(Y) su un dataset locale.

    Args:
        y (array-like): vettore delle etichette binarie (0/1).
        A (array-like): vettore della variabile sensibile binaria (0/1).

    Returns:
        joint (DataFrame): tabella p(A=a, Y=y) normalizzata su tutti gli esempi.
        marginal_y (Series): distribuzione p(Y=y) normalizzata.
    """
    y_arr = np.asarray(y)
    A_arr = np.asarray(A)
    joint = pd.crosstab(A_arr, y_arr, normalize=True)
    marginal_y = pd.Series(y_arr).value_counts(normalize=True)
    return joint, marginal_y

# --- Funzione per calcolare i pesi con Statistical Parity ---
def compute_weights_statistical_parity(y, A):
    """
    Calcola i pesi di reweighting secondo la Statistical Parity.
    w(a,y) = p(Y=y) / p(A=a, Y=y)
    Usa compute_local_stats internamente.
    """
    joint, marginal_y = compute_local_stats(y, A)
    print("Joint distribution (p(A, Y)):\n", joint)
    print("Marginal distribution (p(Y)):\n", marginal_y)
    y_arr = np.asarray(y)
    A_arr = np.asarray(A)
    weights = [marginal_y[y_val] / joint.loc[a_val, y_val] for a_val, y_val in zip(A_arr, y_arr)]
    return weights

# --- Funzione per calcolare la Statistical Parity Difference ---
def compute_statistical_parity(y_hat, A):
    """
    Calcola la Statistical Parity Difference su predizioni y_hat e attributo sensibile A.
    """
    y_hat_arr = np.asarray(y_hat)
    A_arr = np.asarray(A)
    rate_0 = y_hat_arr[A_arr == 0].mean()
    rate_1 = y_hat_arr[A_arr == 1].mean()
    return abs(rate_0 - rate_1)

# --- 1. Caricamento del dataset COMPAS da Hugging Face ---
dataset = load_dataset("imodels/compas-recidivism")
df = pd.DataFrame(dataset['train'])

# --- 2. Divisione in train/test (80-20) ---
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['is_recid']
)
# Estrazione feature, etichetta e attributo sensibile
label_col = 'is_recid'
sensitive_col = 'sex:Female'
X_train = train_df.drop(columns=[label_col, sensitive_col])
y_train = train_df[label_col]
A_train = train_df[sensitive_col].astype(int)
X_test = test_df.drop(columns=[label_col, sensitive_col])
y_test = test_df[label_col]
A_test = test_df[sensitive_col].astype(int)

# --- 3. Calcolo delle statistiche locali sul train set ---
joint_train, marginal_y_train = compute_local_stats(y_train, A_train)
print("Statistiche locali (train):")
print(joint_train)
print(marginal_y_train)

# --- 4. Selezione strategia e generazione dei pesi ---
strategy = 'stat_parity'  # 'stat_parity', 'equal_opp' o 'none'
if strategy == 'stat_parity':
    weights_train = compute_weights_statistical_parity(y_train, A_train)
    print("Usando pesi per Statistical Parity")
elif strategy == 'equal_opp':
    weights_train = None  # placeholder
    print("Equal Opportunity non implementata in questo esempio")
elif strategy == 'none':
    weights_train = None
    print("Baseline: nessun reweighting applicato")
else:
    raise ValueError(f"Strategy '{strategy}' non riconosciuta")

# --- 5. Addestramento sul train set ---
clf = LogisticRegression(solver='liblinear', max_iter=1000)
if weights_train is not None:
    clf.fit(X_train, y_train, sample_weight=weights_train)
else:
    clf.fit(X_train, y_train)

# --- 6. Valutazione sul test set ---
y_pred = clf.predict(X_test)
print("Classification report (test set):")
print(classification_report(y_test, y_pred, target_names=['No Recidiva', 'Recidiva']))

# --- 7. Calcolo e stampa della Statistical Parity Difference ---
sp_diff = compute_statistical_parity(y_pred, A_test)
print(f"Statistical Parity Difference (test): {sp_diff:.4f}")
