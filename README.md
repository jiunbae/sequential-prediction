## Split train, test dataset from raw data

```
python scripts/split.py --data ./data/raw.csv --output ./data
```

`scripts/split.py` generate `train.csv` and `test.csv` which drop target labels from raw data. Basically, `train.csv` contains columns(`'Month, Day, Hour, Quarter, P1(DayOfWeek), Demand'`) where year is 2017.

## Data augmentation

```
python scripts/augmentation.py --train ./data/train.csv --output ./data/train-augmented.csv --repeat 8
```

`scripts/augmentation.py` generated `train-augmented.csv` for training.

## Run

```
python main.py --train ./data/train-augmented.csv --test ./data/test.csv --output ./results --input-size 32 --hidden-size 128
```
Main scripts create `./results` directory which includes `model.h5` and `prediction.csv`.


## Model detail

#### Hyperparameters
- hidden_size: 128
- input_size: 32

![Model](../assets/model.png)
