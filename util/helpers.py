from os.path import join

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    Trainer,
    EarlyStoppingCallback,
)


def load_dataset(path, encoding="utf-8", shuffle=True):
    df = pd.read_csv(path, encoding=encoding)
    df.drop_duplicates(inplace=True)
    if shuffle:
        df = df.sample(frac=1, random_state=9).reset_index(drop=True)
    return df


def load_dataset_with_features(dataset, features_root_path='data/features', ignore_stds=True, english_features=True):
    df = load_dataset(f'data/{dataset}_set.csv')

    # load handcrafted features
    df_features_complexity_de = pd.read_csv(join(features_root_path, f'features_{dataset}_complexity_de.csv'))
    df_features_complexity_de.drop(columns=['Sentence DE', 'sentence_encoded'], inplace=True)

    df_features_complexity_en = pd.read_csv(join(features_root_path, f'features_{dataset}_complexity_en.csv'))
    df_features_complexity_en.drop(columns=['Sentence EN', 'sentence_encoded'], inplace=True)
    df_features_complexity_en.columns = ['ID'] + [c + '_en' for c in df_features_complexity_en.columns[1:]]

    df_features_readability_de = pd.read_csv(join(features_root_path, f'features_{dataset}_readability_de.csv'))
    df_features_readability_de.drop(columns=['Sentence DE'], inplace=True)

    df_features_readability_en = pd.read_csv(join(features_root_path, f'features_{dataset}_readability_en.csv'))
    df_features_readability_en.drop(columns=['Sentence EN'], inplace=True)
    df_features_readability_en.columns = ['ID'] + [c + '_en' for c in df_features_readability_en.columns[1:]]

    df_merged = df.merge(df_features_complexity_de, on='ID')
    if english_features:
        df_merged = df_merged.merge(df_features_complexity_en, on='ID')
    df_merged = df_merged.merge(df_features_readability_de, on='ID')
    if english_features:
        df_merged = df_merged.merge(df_features_readability_en, on='ID')

    # extract all columns without stdev
    if ignore_stds:
        columns_wo_stdev = [c for c in df_merged.columns if 'stdev' not in c]
    else:
        columns_wo_stdev = df_merged.columns
    # ignore some columns
    ignore_columns = ['paragraphs_en', 'sentences_per_paragraph_en', 'paragraphs', 'sentences_per_paragraph']
    columns_wo_stdev = [c for c in columns_wo_stdev if c not in ignore_columns]

    df_merged = df_merged[columns_wo_stdev]

    # add some of our own features
    df_merged['max_word_length'] = df_merged['Sentence DE'].apply(lambda x: max([len(w) for w in x.split()]))
    if english_features:
        df_merged['max_word_length_en'] = df_merged['Sentence EN'].apply(lambda x: max([len(w) for w in x.split()]))
    for i in range(5, 10):
        df_merged['num_word_longer_than_' + str(i)] = df_merged['Sentence DE'].apply(
            lambda x: sum([len(w) > i for w in x.split()]))
        if english_features:
            df_merged['num_word_longer_than_' + str(i) + '_en'] = df_merged['Sentence EN'].apply(
                lambda x: sum([len(w) > i for w in x.split()]))

    feature_columns = df_merged.columns.to_list()[df_merged.columns.to_list().index('Sentence EN') + 1:]

    return df_merged, feature_columns


# ========================
# CLASSES
# ========================
class TCCDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return tensor
        item = {key: val[idx].clone().detach() for key, val in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


class OptimizedESCallback(EarlyStoppingCallback):
    def __init__(self, patience, initial_steps_wo_save):
        super().__init__(early_stopping_patience=patience)
        self.initial_steps_wo_save = initial_steps_wo_save

    def check_metric_value(self, args, state, control, metric_value):
        super().check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter == 0:
            control.should_save = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.global_step < self.initial_steps_wo_save:
            return
        super().on_evaluate(args, state, control, metrics, **kwargs)


# ========================
# COMPUTATION METHODS
# ========================


def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
    }


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    rmse = mean_squared_error(labels, logits, squared=False)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)

    return {
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
    }


def get_hugging_face_name(name):
    if name == "gbert":
        return "deepset/gbert-large"
    if name == "gelectra":
        return "deepset/gelectra-large"
    if name == "gottbert":
        return "uklfr/gottbert-base"
    if name == "gerpt":
        return "benjamin/gerpt2-large"
    if name == "gpt2_xl_wechsel_german":
        return "malteos/gpt2-xl-wechsel-german"
    if name == "xlm-roberta-large":
        return "xlm-roberta-large"
    if name == "xlm-roberta-xl":
        return "facebook/xlm-roberta-xl"
    if name == "xlm-roberta-xxl":
        return "facebook/xlm-roberta-xxl"
    return ""
