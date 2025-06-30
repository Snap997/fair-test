import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Flower core APIs
from flwr.server.strategy import Strategy
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar
from typing import Dict, Tuple, Union, Optional


def compute_global_reweighting_factors(
    joint_counts: pd.DataFrame,
    marginal_counts: pd.Series,
    total_samples: int
) -> Dict[Tuple[int, int], float]:
    """
    Calcola i fattori di reweighting globali per Statistical Parity.

    Args:
        joint_counts (pd.DataFrame): DataFrame con i conteggi aggregati per (A, Y),
            dove l’indice corrisponde ai valori di A (es. 0/1) e le colonne ai valori di Y (es. 0/1).
        marginal_counts (pd.Series): Serie con i conteggi aggregati per Y.
        total_samples (int): numero totale di esempi usati per l’aggregazione.

    Returns:
        Dict[Tuple[int, int], float]: dizionario dei pesi w[(a_val, y_val)].
    """
    # Passa dai conteggi alle probabilità
    joint_probabilities = joint_counts / total_samples
    marginal_probabilities = marginal_counts / total_samples

    # Costruisci i fattori di reweighting
    global_weights: Dict[Tuple[int, int], float] = {}
    for sensitive_value in joint_probabilities.index:
        for label_value in joint_probabilities.columns:
            a = int(float(sensitive_value))
            y = int(float(label_value))
            weight = marginal_probabilities[y] / joint_probabilities.loc[sensitive_value, label_value]
            global_weights[(a, y)] = float(weight)
    return global_weights

class FairFed(FedAvg):
    """Custom FedAvg strategy applying FairFed reweighting for scikit-learn LR."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = None

    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configura il round: seleziona client e fornisce FitIns incluso il config."""
        # 1. Selezione client
        clients = super().configure_fit(server_round, parameters, client_manager)
        # 2 Calcolo dei pesi globali
        
        for client, fit_ins in clients:
            if self.weights is not None:
                print("Weights: ", self.weights)
                #fit_ins.config["weights"] = json.dumps(self.weights) 
            
        return clients


    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        
        loss, metrics =  super().aggregate_evaluate(server_round, results, failures)
        if "agg_joint" in metrics and "agg_marginal" in metrics and "total" in metrics:
            agg_joint =  pd.DataFrame(json.loads(metrics["agg_joint"]))
            agg_marginal = pd.Series(json.loads(metrics["agg_marginal"]))
            total = int(metrics["total"])
            print("Aggregated joint distribution:", agg_joint)
            print("Aggregated marginal distribution:", agg_marginal)
            print("Total examples:", total)
            if total > 0:
                self.weights = compute_global_reweighting_factors(agg_joint, agg_marginal, total)
                print("Weights computed:", self.weights)
        return loss, metrics

