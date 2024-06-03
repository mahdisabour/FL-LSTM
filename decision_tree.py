from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

import data_handler


if __name__ == "__main__":
    # df = data_handler.load_dataset(
    #     paths=data_handler.get_paths(1, 0, base_path="processed_data_3")
    # )
    df = data_handler.load_dataset(
        paths=["processed_data_3/train1.csv"]
    )
    train, test = train_test_split(df, test_size=0.1)
    x_train, y_train = data_handler.get_x_y(train, binary=True, reshape=False)
    x_test, y_test = data_handler.get_x_y(test, binary=True, reshape=False)

    model = KNeighborsClassifier()
    # model = AdaBoostClassifier(random_state=42)
    # model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    loss = log_loss(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("loss", loss, "\n", "accuracy:", accuracy)

    print(model.get_params(deep=True))
    
