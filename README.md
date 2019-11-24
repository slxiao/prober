# Introduction
`prober` is an automated software bug predictor using machine learning techniques. Specifically, `proper` trains prediction model based on open source datasets, and then predicts whether some code has a bug or not. The prediction is based on various code features, such as McCabe features, Halstead features, etc.


# How it works

- traning data is from [open source dataset](./prober/data.csv)
- training algorithm is based on `tensorflow` and `keras`
- features extraction algorithm is specifc for each feature and each programming language. For java, the feature extraction algorithm can be found in [java](./java)

# How to use
## traning
Install dependencies:
```shell
pip install -r requirements.txt
```

Start traning:
```shell
python -m prober.prober
```

Extract java features:
```shell
gradle build && gradle execute
```
# LISENSE
MIT