import argparse
from typing import Dict, Optional, Tuple
import random

import flwr as fl
import tensorflow as tf
import numpy as np 

import data_handler

NUM_ROUNDS = 10
EPOCHS = 1
BASE_DATASET_PATH = "processed_data_3"


def get_confusion_matrix(model, x_test: list, y_test: list):
    y_predictions = model.predict(x_test)
    y_predictions = np.argmax(y_predictions, axis=1)
    c_matrix = tf.math.confusion_matrix(
        y_test,
        y_predictions,
        num_classes=None,
        weights=None,
        dtype=tf.dtypes.int32,
        name=None
    )
    return c_matrix

def multiClassModel(n_features=78, n_classes=2, time_steps=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(time_steps, n_features)))
    model.add(tf.keras.layers.LSTM(units=30))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
    model.summary()
    return model

def main(n_clients: int, aggregator: str, binary: bool) -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    
    n_classes = 2 if binary else 5 
    model = multiClassModel(n_classes=n_classes)
    function = None

    if aggregator == "FedAvg":
        function = fl.server.strategy.FedAvg
    elif aggregator == "FedAdagrad":
        function = fl.server.strategy.FedAdagrad
    elif aggregator == "FedYogi":
        function = fl.server.strategy.FedYogi
    elif aggregator == "FedAvgM":
        function = fl.server.strategy.FedAvgM
    elif aggregator == "FedTrimmedAvg":
        function = fl.server.strategy.FedTrimmedAvg
        
    if aggregator != "FedProx":
        strategy = function(
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        )
    else:
        strategy = fl.server.strategy.FedProx(
            min_fit_clients=n_clients,
            min_evaluate_clients=n_clients,
            min_available_clients=n_clients,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
            proximal_mu=1
        )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    df = data_handler.load_dataset(
        paths=[f"{BASE_DATASET_PATH}/test.csv"], sample_size=0.1
    )
    x_test, y_test = data_handler.get_x_y(df=df, binary=args.binary)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        result = model.evaluate(x_test, y_test, return_dict=True)
        return result["loss"], {"accuracy": result["accuracy"]}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": EPOCHS,
        "round": server_round,
        "server_rounds": NUM_ROUNDS
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--aggregator",
        type=str,
        default="FedAvg",
        choices=["FedAvg", "FedAdagrad", "FedYogi", "FedAvgM", "FedTrimmedAvg", "FedProx"],
    )
    parser.add_argument(
        "--n-clients",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--binary",
        type=bool,
        default=False
    )
    args = parser.parse_args()

    main(n_clients=args.n_clients, aggregator=args.aggregator, binary=args.binary)
