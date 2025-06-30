"""Fair-Test: A Flower / sklearn app."""

import warnings
import json
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fair_test.task import custom_dataset
from fair_test.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)
from fairlearn.metrics import (
    true_positive_rate,
    MetricFrame,
    equalized_odds_difference
)
from fair_test.metrics import calculate_eor, calculate_eod, calculate_spd, calculate_dpr, compute_weights_statistical_parity, compute_local_stats, apply_global_reweighting


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, reweighting_strategy: str):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.reweighting_strategy = reweighting_strategy

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        #print("Config sample weight:", config)
        weights = None
        if config and "weights" in config and self.reweighting_strategy in ["both", "central"]:
            # If weights are provided in the config, convert them from string to a list
            print("Using weights from config:", config["weights"])
            weightDict = json.loads(config["weights"])
            weights = apply_global_reweighting(weightDict, self.X_train, self.y_train)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.reweighting_strategy == "local":
                print("Using local reweighting strategy")
                weights = compute_weights_statistical_parity(self.y_train, self.X_train[custom_dataset.get_sensitive_features()[0]])
            #print("Computed weights len: {}, weights: {}", len(self.weights), self.weights[:10])

            self.model.fit(self.X_train.values, self.y_train.values, sample_weight=weights)


        return get_model_params(self.model), len(self.X_train.values), {}

    def evaluate(self, parameters, config):
        # --- . Calcolo delle frequenze congiunte e dei pesi ---
        A_train = self.X_train[custom_dataset.get_sensitive_features()[0]]
        A_test = self.X_test[custom_dataset.get_sensitive_features()[0]]
        
        #print("Start evaluation")
        # 1) load params
        set_model_params(self.model, parameters)
        
        # 2) get probability & label predictions
        y_pred_proba = self.model.predict_proba(self.X_test.values)  # for log_loss
        y_pred = self.model.predict(self.X_test.values)              # for metrics

        # 3) compute overall loss & accuracy
        loss = log_loss(self.y_test.values, y_pred_proba, labels=custom_dataset.get_y_classes())
        accuracy = self.model.score(self.X_test.values, self.y_test.values)

        # 4) extract the raw race column as a 1-D array

        #print("A_test sample values:", A_test[:10])
        #print("A_test shape:", A_test.shape)

        # 5) compute group‐wise true positive rate
        
        metric_frame = MetricFrame(
            metrics=true_positive_rate,
            y_true=self.y_test,
            y_pred=y_pred,
            sensitive_features=A_test
        )
        #print("TPR per group:\n", metric_frame.by_group)
        metrics = {}
        metrics["accuracy"] = accuracy
        #metrics["weights"] = weights
        metrics["eor≃1"] = calculate_eor(self.y_test, y_pred, A_test)
        metrics["eod≃0"] = calculate_eod(metric_frame)
        metrics["spd≃0"] = calculate_spd(self.y_test, y_pred, A_test)
        metrics["dpr≃1"] = calculate_dpr(self.y_test, y_pred, A_test)
        metrics["aod≃0"] = float(equalized_odds_difference(y_true=self.y_test,y_pred=y_pred,sensitive_features=A_test,agg="mean"))
        metrics["agg_joint"], metrics["agg_marginal"] = compute_local_stats(self.y_train, A_train)
        metrics["agg_joint"] = json.dumps(metrics["agg_joint"].to_dict())
        metrics["agg_marginal"] = json.dumps(metrics["agg_marginal"].to_dict())
        metrics["total"] = len(self.y_test)

        return loss, len(self.X_test.values), metrics

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    alpha = context.run_config["alpha"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions, alpha)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    reweighting_strategy = context.run_config.get("reweighting-strategy", "none")
    print("Reweighting strategy:", reweighting_strategy)
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, reweighting_strategy).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
