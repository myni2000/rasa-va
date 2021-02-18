# RASA Banking Virtual Assistant

This repo is belonged to our paper "Developing AI virtual assistant to support banking finance guidance using RASA framework for universities". 

## 0. Introduction

We applied RASA platform for building a virtual assitant served for study in universities. Our contribution is adding 3 Vietnamese tokenizers: [PiVy](https://github.com/trungtv/pyvi), [Underthesea](https://github.com/undertheseanlp/underthesea) and [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP).

## 1. Requirements

After cloning this repo, use the following command line for installing nessesary libraries

```
pip install -r requirements.txt
```

## 2. Prepare data for training

Dataset is followed RASA format. Figure it out [here](https://rasa.com/docs/rasa/nlu-training-data/). It should include `nlu.yml`, `domain.yml` and `stories.yml` and be organized as following directory tree:

```
├── tests
├── results
├── actions
├── data
│   ├── <nlu.yml>
│   └── <stories.yml>
├── rasa
├── <domain.yml>
├── models
└── requirements.txt
``` 
We also prepare some samples for you. Prepare your data like these.

## 3. Prepare config.yml

You can use all default RASA tokenizers or Vietnamese tokenizers by changing `config.yml`.

Vietnamese tokenizers are defined as follow:
PyVi: `PyViVietnameseTokenizer`
Underthesea: `VietnameseTokenizer`
VnCoreNLP: `VncoreNLPVietnameseTokenizer`


```
language: vi

pipeline:
# # No configuration for the NLU pipeline was provided. The following default pipeline was used to train your model.
# # If you'd like to customize it, uncomment and adjust the pipeline.
# # See https://rasa.com/docs/rasa/tuning-your-model for more information.
   - name: VietnameseTokenizer <----- HERE
   - name: CountVectorsFeaturizer
   - name: CountVectorsFeaturizer
     analyzer: char_wb
     min_ngram: 1
     max_ngram: 4
#   - name: LanguageModelFeaturizer
#     model_name: "bert"
#     model_weights: "bert-base-multilingual-uncased"
   - name: DIETClassifier
     batch_size: [16, 64]
     epochs: 100
   - name: EntitySynonymMapper
   - name: ResponseSelector
     epochs: 100
   - name: DIETClassifier
     threshold: 0.3
     ambiguity_threshold: 0.1
     batch_size: [8,16]

policies:
  - name: TEDPolicy
    max_history: 5
    epochs: 200
    batch_size: 20
    max_training_samples: 600
```

## 4. Command line

### Training

Train both Core and NLP:

```
python -m rasa train
```

Train NLP:

```
python -m rasa train nlu
```

Train Core:

```
python -m rasa train core
```

### Testing

Conservation with chatbot

```
python -m rasa shell
```

Or check more command line [here](https://rasa.com/docs/rasa/command-line-interface/)

Feel free for asking if you face issues when implementing our code.
