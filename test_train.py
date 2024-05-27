import random

import tensorflow as tf

import data_handler

def multiClassModel(n_features=43, n_classes=2, time_steps=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(time_steps, n_features)))
    model.add(tf.keras.layers.LSTM(units=30))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model




if __name__ == "__main__":
    model = multiClassModel(n_features=78)
    # Load a subset of CICIDS2017 to simulate the local data partition
    csv_paths = data_handler.get_paths(1, 0, "processed_data_1/")
    df = data_handler.load_dataset(paths=csv_paths)
    x_train, y_train, x_test, y_test = data_handler.split_test_train(df=df)

    # Convert DataFrame to NumPy array
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values

    # Reshape data to include time_steps dimension
    time_steps = 1
    x_train = x_train.reshape((x_train.shape[0], time_steps, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], time_steps, x_test.shape[1]))

    batch_size: int = 32
    epochs: int = 2

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )

    results = {
        "loss": history.history["loss"][0],
        "accuracy": history.history["accuracy"][0],
        "val_loss": history.history["val_loss"][0],
        "val_accuracy": history.history["val_accuracy"][0],
    }
    print(f"{results=}")

    df = data_handler.load_dataset(paths=["processed_data_1/train1.csv"])
    _, _, x_test, y_test = data_handler.split_test_train(df=df, test_size=0.9)

    # Convert DataFrame to NumPy array
    x_test = x_test.values
    y_test = y_test.values

    # Reshape x_test to have a time_steps dimension
    time_steps = 1
    x_test = x_test.reshape((x_test.shape[0], time_steps, x_test.shape[1]))

    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss, accuracy)
