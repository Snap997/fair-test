from typing import List, Tuple, Sequence, Dict
from flwr.common import Metrics
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    true_positive_rate, MetricFrame,
    equal_opportunity_ratio, 
    equalized_odds_difference,
    demographic_parity_difference, 
    demographic_parity_ratio
)


def calculate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    m = Metrics()
    for num_examples, metric in metrics:
        # Add the number of examples to each metric
        for key, value in metric.items():
            if key not in m:
                m[key] = 0.0
            if isinstance(value, str):
                m[key] = value  
            else:
                m[key] += num_examples * value

    # Normalize the metrics by the total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    for key in m:
        if isinstance(m[key], str):
            continue
        m[key] /= total_examples

    return m

def calculate_metric(metrics: List[Tuple[int, Metrics]], metric_name: str) -> Metrics:
    values = [num_examples * m[metric_name] for num_examples, m in metrics if metric_name in m]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return sum(values) / total_examples


def calculate_eor(y_test, y_pred, A_test) -> float:
    # 6) compute equal opportunity ratio
    ratio = equal_opportunity_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    )
    print(f"Equal Opportunity Ratio: {ratio:.3f}")
    return float(ratio)

def calculate_eod(metric_frame) -> float:
    # Calcola Equal Opportunity Difference
    tpr_values = metric_frame.by_group
    eod = tpr_values.max() - tpr_values.min()
    return float(eod)

def calculate_spd(y_test, y_pred, A_test) -> float:
    # Calcola SPD e DPR
    return float(demographic_parity_difference(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    ))

def calculate_dpr(y_test, y_pred, A_test) -> float:
    return float(demographic_parity_ratio(
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=A_test
    ))


def compute_weights_statistical_parity(y, A):
    """
    Calcola i pesi di reweighting secondo la Statistical Parity.
    w(a,y) = p(Y=y) / p(A=a, Y=y)

    Args:
        y (array-like): vettore delle etichette binarie (0/1).
        A (array-like): vettore della variabile sensibile binaria (0/1).

    Returns:
        List[float]: peso per ciascuna istanza.
    """
    y_series = pd.Series(y)
    A_series = pd.Series(A)
    # Calcola distribuzioni
    joint = pd.crosstab(A_series, y_series, normalize=True)
    marginal_y = y_series.value_counts(normalize=True)
    # Genera pesi
    weights = []
    for a_val, y_val in zip(A_series, y_series):
        w = marginal_y[y_val] / joint.loc[a_val, y_val]
        weights.append(w)
    return weights

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
    #print("Joint distribution (p(A, Y)):\n", joint)
    #print("Marginal distribution (p(Y)):\n", marginal_y)
    return joint, marginal_y


def apply_global_reweighting(
    global_weights: Dict[Tuple[int, int], float],
    A: Sequence[int],
    y: Sequence[int]
) -> List[float]:
    """
    Applica i fattori di reweighting globali (Statistical Parity) a un dataset
    restituendo la lista dei pesi per ciascuna istanza.

    Args:
        global_weights: dizionario {(a, y): w(a,y)} calcolato da
            compute_global_reweighting_factors.
        A: vettore (o lista/Series) dei valori dell'attributo sensibile (0/1).
        y: vettore (o lista/Series) delle etichette (0/1).

    Returns:
        weights: lista di float di lunghezza len(A), con il peso di ogni istanza.
    """
    if len(A) != len(y):
        raise ValueError("A e y devono avere la stessa lunghezza")

    weights: List[float] = []
    for a_val, y_val in zip(A, y):
        key = (int(a_val), int(y_val))
        try:
            w = global_weights[key]
        except KeyError:
            raise KeyError(f"Nessun peso definito per la combinazione A={a_val}, y={y_val}")
        weights.append(w)
    return weights