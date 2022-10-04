# GermEval 2022
Repository containing the experiments described in our paper for the 
[GermEval 2022 challenge](https://aclanthology.org/2022.germeval-1.1/): 
"Automatic Readability Assessment of German Sentences with Transformer Ensembles" 
([available online](https://aclanthology.org/2022.germeval-1.10/)).

## Dataset 
We translated the given German dataset to English during preprocessing.
The resulting training, validation and test split is available in the folder [data](./data).

Additionally we created some handcrafted readability features for German as well as English.
The scripts used to generate these can be found in [data_processing](./data_processing).
Their corresponding splits can be found in [data/features](./data/features).

## Experiments 
To evaluate the ability of a model (ensemble) to predict the complexity of a sentence, we conducted experiments with the following parameters:

- `BOOTSTRAP_SIZE`: number of random samples with replacement from a given population of models
- `MAX_ENSEMBLE_SIZE`: max amount of members within one ensemble, used to iterate to
- `ENSEMBLE_POOL_SIZE`: model population size

The following experiments are available in [experiments](./experiments):

- `ensemble_size_gbert.py`: evaluating GBERT ensembles
- `ensemble_size_wechsel.py`: evaluating GPT-2-Wechsel ensembles
- `ensemble_size_wechsel_gbert.py`: evalualting mixed ensembles

The submitted best ensemble generation can be found in [submissions/sub_ensemble_wechsel_gbert.py](./submissions/sub_ensemble_wechsel_gbert.py)

## Installation

A current Anaconda/Miniconda environment is required.

To install the given conda environment use the following command:

```sh
conda env create -f environment.yml
```

## Academic citation

Please use the following citation when referencing this software:

Patrick Gustav Blaneck, Tobias Bornheim, Niklas Grieger, and Stephan Bialonski. 2022. [Automatic Readability Assessment of German Sentences with Transformer Ensembles](https://aclanthology.org/2022.germeval-1.10). In Proceedings of the GermEval 2022 Workshop on Text Complexity Assessment of German Text, pages 57–62, Potsdam, Germany. Association for Computational Linguistics.

```
@inproceedings{blaneck2022,
    title = "Automatic Readability Assessment of {G}erman Sentences with Transformer Ensembles",
    author = "Blaneck, Patrick Gustav  and
      Bornheim, Tobias  and
      Grieger, Niklas  and
      Bialonski, Stephan",
    booktitle = "Proceedings of the GermEval 2022 Workshop on Text Complexity Assessment of German Text",
    month = sep,
    year = "2022",
    address = "Potsdam, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.germeval-1.10",
    pages = "57--62",
}
```
