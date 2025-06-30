"""Fair-Test: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from strategy.fair_fed import FairFed
from fair_test.task import get_model, get_model_params, set_initial_params
from fair_test.metrics import calculate_metrics
from flwr.server.strategy.fedavg import FedAvg

def test(num_round: int) -> dict:
    print("on fit config function called with num_round:", num_round)
    return {"round": num_round}

def test_evaluate(num_round: int) -> dict:
    print("on evaluate config function called with num_round:", num_round)
    return {"round": num_round}

def metricsTest(metrics):
    print("Metrics aggregation function called with metrics:", metrics)
    return {}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = FairFed(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        inplace=False,
        evaluate_metrics_aggregation_fn=calculate_metrics,
        #on_fit_config_fn=test,
        #on_evaluate_config_fn=test_evaluate,
        #fit_metrics_aggregation_fn=metricsTest
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
