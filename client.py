import argparse
import os
import random

import tensorflow as tf

import flwr as fl

import data_handler

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_DATASET_PATH = "processed_data_3"
LEARNING_RATE = 0.0001


def multiClassModel(n_features=78, n_classes=2, time_steps=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(time_steps, n_features)))
    model.add(tf.keras.layers.LSTM(units=30))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


# Define Flower client
class CICIDSClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        round: int = config["round"]
        server_rounds: int = config["server_rounds"]

        step = len(self.x_train) // server_rounds

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train[(round-1)*step:round*step],
            self.y_train[(round-1)*step:round*step],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--n-clients",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of CICIDS2017 to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--binary",
        type=bool,
        default=False
    )
    args = parser.parse_args()

    n_classes = 2 if args.binary else 5
    # Load and compile Keras model
    model = multiClassModel(n_classes=n_classes)

    # Load a subset of CICIDS2017 to simulate the local data partition
    csv_paths = data_handler.get_paths(args.n_clients, args.client_id, base_path=f"{BASE_DATASET_PATH}/")
    df = data_handler.load_dataset(paths=csv_paths)
    x_train, y_train = data_handler.get_x_y(df=df, binary=args.binary)
    df = data_handler.load_dataset(paths=[f"{BASE_DATASET_PATH}/test.csv"])
    x_test, y_test = data_handler.get_x_y(df=df, binary=args.binary)

    # Start Flower client
    client = CICIDSClient(model, x_train, y_train, x_test, y_test).to_client()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


if __name__ == "__main__":
    main()
