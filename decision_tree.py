from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import log_loss, accuracy_score

import data_handler


if __name__ == "__main__":
    all_paths = data_handler.get_paths(
        1, 0, "data/MachineLearningCSV/MachineLearningCVE"
    )
    df_train = data_handler.load_dataset(
        paths=all_paths,
        preprocess=True
    )
    df_test = data_handler.load_dataset(
        paths=[all_paths[-1]],
        preprocess=True
    )

    x_train, y_train = data_handler.get_x_y(df_train, binary=True, reshape=False)
    x_test, y_test = data_handler.get_x_y(df_test, binary=True, reshape=False)

    model = XGBClassifier()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    loss = log_loss(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("loss", loss, "\n", "accuracy:", accuracy)