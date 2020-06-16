## Split train, test dataset from raw data

```
python scripts/split.py --data ./data/raw.csv --output ./data
```

`scripts/split.py` generate `train.csv` and `test.csv` which drop target labels from raw data. Basically, `train.csv` contains columns(`'Month, Day, Hour, Quarter, P1(DayOfWeek), Demand'`) where year is 2017.

## Run
```
python main.py --train ./data/train.csv --test ./data/test.csv --output ./results
```
Main scripts create `./results` directory which includes `model.h5` and `prediction.csv`.
