import hashlib
import os
import sys
from os.path import realpath, join, dirname

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    set_seed,
)

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from util.helpers import (
    load_dataset_with_features, get_hugging_face_name, TCCDataset, RegressionTrainer, compute_metrics_for_regression,
    OptimizedESCallback
)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MAX_ENSEMBLE_SIZE = 100
SAVE_AFTER_N_MODELS = 10
MODEL_NAMES = ['gpt2_xl_wechsel_german', 'gbert']  # ['gbert', 'gelectra', 'gottbert', 'gerpt']
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
N_EVAL_STEPS = 28

EXPERIMENT_NAME = 'sub_ensemble_wechsel_gbert'
EXPERIMENT_DIR = f'cache/{EXPERIMENT_NAME}'

os.makedirs(f'{EXPERIMENT_DIR}/models/wechsel', exist_ok=True)
os.makedirs(f'{EXPERIMENT_DIR}/models/mlp', exist_ok=True)
os.makedirs(f'{EXPERIMENT_DIR}/predictions', exist_ok=True)

df_train, feature_columns_train = load_dataset_with_features('training')
df_valid, feature_columns_valid = load_dataset_with_features('valid')
df_test, feature_columns_test = load_dataset_with_features('test')
assert np.all(np.array(feature_columns_train) == np.array(feature_columns_valid))
assert np.all(np.array(feature_columns_train) == np.array(feature_columns_test))


def get_predictions(
        n_epochs=100,
        n_log_steps=10,
):
    # store predictions in dataframe
    # columns: Sentence, Prediction of Model 1, Prediction of Model 2, ...
    df_predictions_valid = {model_name: df_valid[['ID', 'Sentence DE']].copy() for model_name in MODEL_NAMES}
    df_predictions_test = {model_name: df_test[['ID', 'Sentence DE']].copy() for model_name in MODEL_NAMES}

    n_trained_models = {model_name: 0 for model_name in MODEL_NAMES}
    while np.all([n_models < MAX_ENSEMBLE_SIZE for n_models in n_trained_models.values()]):
        for inner_model_name in MODEL_NAMES:
            # get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(get_hugging_face_name(inner_model_name))

            # if inner_model_name is either gpt2_xl_wechsel_german or gerpt2 we need to set the padding token
            if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

            X_valid = df_valid['Sentence DE'].values
            X_test = df_test['Sentence DE'].values
            X_valid_features = df_valid[feature_columns_train].values
            X_test_features = df_test[feature_columns_train].values

            # normalize data with StandardScaler
            scaler = StandardScaler()
            scaler.fit(df_train[feature_columns_train].values)
            X_valid_features = scaler.transform(X_valid_features)
            X_test_features = scaler.transform(X_test_features)

            # tokenize
            if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                X_valid = np.array([f'{tokenizer.bos_token} {x} {tokenizer.eos_token}' for x in X_valid])
                X_test = np.array([f'{tokenizer.bos_token} {x} {tokenizer.eos_token}' for x in X_test])
            tokens_valid = tokenizer(X_valid.tolist(), padding='max_length', return_tensors='pt', truncation=True,
                                     max_length=128)
            tokens_test = tokenizer(X_test.tolist(), padding='max_length', return_tensors='pt', truncation=True,
                                    max_length=128)

            for k in range(n_trained_models[inner_model_name],
                           n_trained_models[inner_model_name] + SAVE_AFTER_N_MODELS):
                df_early_stopping = df_train.sample(frac=0.075, random_state=k)
                df_training = df_train.drop(
                    df_train[df_train['Sentence DE'].isin(df_early_stopping['Sentence DE'])].index
                )
                df_training = df_training.fillna(df_training.mean(numeric_only=True))
                df_early_stopping = df_early_stopping.fillna(df_training.mean(numeric_only=True))

                X_early_stopping = df_early_stopping['Sentence DE'].values
                X_early_stopping_features = df_early_stopping[feature_columns_train].values
                y_early_stopping = df_early_stopping['MOS'].values

                X_training = df_training['Sentence DE'].values
                X_training_features = df_training[feature_columns_train].values
                y_training = df_training['MOS'].values

                # tokenize
                if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                    X_early_stopping = np.array(
                        [f'{tokenizer.bos_token} {x} {tokenizer.eos_token}' for x in X_early_stopping])
                tokens_early_stopping = tokenizer(X_early_stopping.tolist(), padding='max_length', return_tensors='pt',
                                                  truncation=True, max_length=128, )

                if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                    X_training = np.array([f'{tokenizer.bos_token} {x} {tokenizer.eos_token}' for x in X_training])
                tokens_training = tokenizer(X_training.tolist(), padding='max_length', return_tensors='pt',
                                            truncation=True, max_length=128, )

                hash = (
                        hashlib.sha256(
                            pd.util.hash_pandas_object(df_training['ID'], index=True).values
                        ).hexdigest()
                        + '_'
                        + get_hugging_face_name(inner_model_name)[
                          get_hugging_face_name(inner_model_name).find('/') + 1:]
                )

                # load transformer model and, if necessary, train it
                try:
                    print(f'{EXPERIMENT_DIR}/models/{inner_model_name}/{hash}')
                    model = AutoModelForSequenceClassification.from_pretrained(
                        f'{EXPERIMENT_DIR}/models/{inner_model_name}/{hash}', local_files_only=True, num_labels=1
                    )

                    # if inner_model_name is either gpt2_xl_wechsel_german or gerpt2 we need to set the padding token
                    if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                        model.resize_token_embeddings(len(tokenizer))
                        model.config.pad_token_id = tokenizer.pad_token_id
                except EnvironmentError:
                    # create training dataset
                    early_stopping_dataset = TCCDataset(tokens_early_stopping, y_early_stopping)
                    training_dataset = TCCDataset(tokens_training, y_training)

                    training_args = TrainingArguments(
                        output_dir=f'{EXPERIMENT_DIR}/{inner_model_name}_trainer/',
                        num_train_epochs=n_epochs,
                        per_device_train_batch_size=TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=TEST_BATCH_SIZE,
                        warmup_ratio=0.3,
                        learning_rate=5e-5,
                        no_cuda=False,
                        metric_for_best_model='root_mean_squared_error',
                        greater_is_better=False,
                        load_best_model_at_end=True,
                        # we never want to save a model through this function, but the parameter must be set, because of load_best_model_at_end=True
                        save_steps=N_EVAL_STEPS * 100_000,
                        save_total_limit=1,  # can be 1, because we only save, when we find a better model
                        eval_steps=N_EVAL_STEPS,
                        evaluation_strategy='steps',
                        seed=k,
                        logging_steps=n_log_steps,
                        logging_dir=f'{EXPERIMENT_DIR}/logs/member_{k}',
                        logging_strategy='steps',
                    )

                    set_seed(training_args.seed)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        get_hugging_face_name(inner_model_name), num_labels=1
                    )

                    # if inner_model_name is either gpt2_xl_wechsel_german or gerpt2 we need to set the padding token
                    if inner_model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                        model.resize_token_embeddings(len(tokenizer))
                        model.config.pad_token_id = tokenizer.pad_token_id

                    trainer = RegressionTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=training_dataset,
                        eval_dataset=early_stopping_dataset,
                        compute_metrics=compute_metrics_for_regression,
                        callbacks=[OptimizedESCallback(patience=5, initial_steps_wo_save=300)],
                    )
                    # training
                    trainer.train()

                    # save model
                    model.save_pretrained(f'{EXPERIMENT_DIR}/models/{inner_model_name}/{hash}')

                # load MLP model and, if necessary, train it
                try:
                    mlp = load_model(f'{EXPERIMENT_DIR}/models/mlp/{hash}_mlp.h5')
                except Exception:
                    hidden_state_train = extract_hidden_state(inner_model_name, model, tokens_training)
                    hidden_state_early_stopping = extract_hidden_state(inner_model_name, model, tokens_early_stopping)

                    np.random.seed(k)
                    mlp = Sequential(
                        [
                            Input(shape=(model.config.hidden_size + len(feature_columns_train),), name='input'),
                            Dense(model.config.hidden_size, activation='relu', name='layer1'),
                            Dense(1, activation='linear', name='layer2'),
                        ]
                    )
                    mlp.compile(
                        optimizer='rmsprop',
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.RootMeanSquaredError()],
                    )
                    es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1, patience=100)
                    mc = ModelCheckpoint(f'{EXPERIMENT_DIR}/models/mlp/{hash}_mlp.h5',
                                         monitor='val_root_mean_squared_error',
                                         mode='min', verbose=1, save_best_only=True)

                    # normalize data with StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(X_training_features)
                    X_train_features = scaler.transform(X_training_features)
                    X_es_features = scaler.transform(X_early_stopping_features)

                    X_train = np.concatenate((hidden_state_train.detach().numpy(), X_train_features), axis=1)
                    X_es = np.concatenate((hidden_state_early_stopping.detach().numpy(), X_es_features), axis=1)

                    mlp.fit(X_train, y_training,
                            validation_data=(X_es, y_early_stopping),
                            batch_size=TRAIN_BATCH_SIZE,
                            epochs=5000, callbacks=[es, mc])
                    mlp = load_model(f'{EXPERIMENT_DIR}/models/mlp/{hash}_mlp.h5')

                # load hidden states of wechsel model for test data
                hidden_state_valid = extract_hidden_state(inner_model_name, model, tokens_valid)
                hidden_state_test = extract_hidden_state(inner_model_name, model, tokens_test)
                X_valid_with_features = np.concatenate((hidden_state_valid.detach().numpy(), X_valid_features), axis=1)
                X_test_with_features = np.concatenate((hidden_state_test.detach().numpy(), X_test_features), axis=1)
                # predict MLP on validation and test sets
                prediction_valid = mlp.predict(X_valid_with_features, batch_size=TEST_BATCH_SIZE)
                prediction_test = mlp.predict(X_test_with_features, batch_size=TEST_BATCH_SIZE)

                df_predictions_valid[inner_model_name][f'{inner_model_name}_prediction_{k}'] = prediction_valid
                df_predictions_test[inner_model_name][f'{inner_model_name}_prediction_{k}'] = prediction_test

                n_trained_models[inner_model_name] += 1

            df_predictions_valid[inner_model_name].to_csv(
                f'{EXPERIMENT_DIR}/predictions/predictions_{inner_model_name}_valid_{n_trained_models[inner_model_name]}.csv')
            df_predictions_test[inner_model_name].to_csv(
                f'{EXPERIMENT_DIR}/predictions/predictions_{inner_model_name}_test_{n_trained_models[inner_model_name]}.csv')
            print(f'Saved predictions for {inner_model_name}_{n_trained_models[inner_model_name]}')

    return df_predictions_valid, df_predictions_test


def extract_hidden_state(model_name, model, tokens, batch_size=16):
    last_last_hidden_state = torch.zeros((len(tokens.input_ids), model.config.hidden_size))
    model = model.cuda().eval()
    with torch.no_grad():
        for i in range(0, len(tokens.input_ids), batch_size):
            if i + batch_size > len(tokens.input_ids):
                input_i = tokens.input_ids[i:]
            else:
                input_i = tokens.input_ids[i:i + batch_size]
            output = model(input_i.cuda(), output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1].cpu()
            if model_name in ['gpt2_xl_wechsel_german', 'gpt2']:
                idx_last_token = torch.ne(input_i, model.config.pad_token_id).sum(-1) - 1
            else:
                idx_last_token = torch.zeros(len(input_i)).long()
            last_last_hidden_state[i:i + len(idx_last_token)] = last_hidden_state[
                torch.arange(len(idx_last_token)), idx_last_token]
    return last_last_hidden_state


# fill na with mean of columns
df_valid = df_valid.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_train.mean(numeric_only=True))

pool_predictions_valid, pool_predictions_test = get_predictions()

# save predictions
for model_name in MODEL_NAMES:
    pool_predictions_valid[model_name].to_csv(f'{EXPERIMENT_DIR}/predictions/predictions_{model_name}_valid_all.csv')
    pool_predictions_test[model_name].to_csv(f'{EXPERIMENT_DIR}/predictions/predictions_{model_name}_test_all.csv')
