# Federated Learning using LSTM and Flower

## Download dataset 
you should download CICIDS 2017 dataset first
[dataset](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

also you can download dataset via this code snippet.
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

print("Path to dataset files:", path)
```

## Run data handler

you should run this command before starting core concept. 

```shell
python data_handler.py
```
consider that you must change dataset path in dataset_handler.py code.

```python
if __name__ == "__main__":
    dataset_path = "data"
    paths = get_paths(1, 0, dataset_path)
    df = load_dataset(paths=paths)
    print(sorted(df[" Label"].unique()))
    df = df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(df, random_state=42, test_size=0.1)
    test.to_csv("processed_data_3/test.csv", index=False)
    split_and_save_data_frame(
        df=train,
        output_path="processed_data_3/",
        file_counts=8
    )
```

## RUN
```shell
./run.sh AGGREGATOR=$aggregator N_CLIENTS=$n_clients BINARY=true
```
