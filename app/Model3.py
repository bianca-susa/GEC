import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    MT5ForConditionalGeneration
)

import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

pd.set_option('display.max_colwidth', None)

# Load data
test_df = pd.read_csv('C:/Users/bibis/PycharmProjects/GEC/app/data/test.csv')
train_df = pd.read_csv('C:/Users/bibis/PycharmProjects/GEC/app/data/train.csv')

# Initialize model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def calc_token_len(example):
    return len(tokenizer(example).input_ids)


from sklearn.model_selection import train_test_split

test_df['input_token_len'] = test_df['input'].apply(calc_token_len)

from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

from torch.utils.data import Dataset, DataLoader


class GrammarDataset(Dataset):
    def __init__(self, dataset, tokenizer, print_text=False):
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.print_text = print_text
        self.max_len = 64

    def __len__(self):
        return len(self.dataset)

    def tokenize_data(self, example):
        input_, target_ = example['input'], example['output']

        # tokenize inputs
        tokenized_inputs = tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
                                     max_length=self.max_len,
                                     return_attention_mask=True)

        tokenized_targets = tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
                                      max_length=self.max_len,
                                      return_attention_mask=True)

        inputs = {"input_ids": tokenized_inputs['input_ids'],
                  "attention_mask": tokenized_inputs['attention_mask'],
                  "labels": tokenized_targets['input_ids']
                  }

        return inputs

    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        if self.print_text:
            for k in inputs.keys():
                print(k, len(inputs[k]))

        return inputs


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')

# Define training arguments
batch_size = 16
args = Seq2SeqTrainingArguments(output_dir="/content/drive/MyDrive/c4_200m/weights",
                                evaluation_strategy="steps",
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                learning_rate=2e-5,
                                num_train_epochs=1,
                                weight_decay=0.01,
                                save_total_limit=2,
                                predict_with_generate=True,
                                fp16=True,
                                gradient_accumulation_steps=6,
                                eval_steps=500,
                                save_steps=500,
                                load_best_model_at_end=True,
                                logging_dir="/logs",
                                report_to="wandb")

import numpy as np
from evaluate import load

bleu_metric = load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Prepare BLEU references and hypotheses
    references = [[ref.split()] for ref in decoded_labels]
    hypotheses = [pred.split() for pred in decoded_preds]

    # Compute BLEU score
    bleu_results = bleu_metric.compute(predictions=hypotheses, references=references)
    bleu_score = bleu_results["bleu"] * 100

    # Compute Precision, Recall, and F0.5 Score
    precision_list = []
    recall_list = []

    for pred, ref in zip(decoded_preds, decoded_labels):
        pred_tokens = tokenizer.encode(pred, add_special_tokens=False)
        ref_tokens = tokenizer.encode(ref, add_special_tokens=False)

        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)

        true_positives = len(pred_set & ref_set)

        precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
        recall = true_positives / len(ref_set) if len(ref_set) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)

    beta = 0.5
    f0_5 = (1 + beta**2) * (mean_precision * mean_recall) / ((beta**2 * mean_precision) + mean_recall) if (mean_precision + mean_recall) > 0 else 0

    return {"precision": round(mean_precision * 100, 4),
            "recall": round(mean_recall * 100, 4),
            "f0_5": round(f0_5 * 100, 4),
            "bleu": round(bleu_score, 4)}


trainer = Seq2SeqTrainer(model=model,
                         args=args,
                         train_dataset=GrammarDataset(train_dataset, tokenizer),
                         eval_dataset=GrammarDataset(test_dataset, tokenizer),
                         tokenizer=tokenizer,
                         data_collator=data_collator,
                         compute_metrics=compute_metrics)

trainer.train()
eval_results = trainer.evaluate()

# Display F0.5 score and BLEU score
print(f"F0.5 Score: {eval_results['eval_f0_5']:.4f}")
print(f"BLEU Score: {eval_results['eval_bleu']:.4f}")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'deep-learning-analytics/GrammarCorrector'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def correct_grammar(input_text, num_return_sequences):
    batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(
        torch_device)
    translated = model.generate(**batch, max_length=64, num_beams=4, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

text = 'Săi ținem pumnii să se distreze..'
print(correct_grammar(text, num_return_sequences=2))

