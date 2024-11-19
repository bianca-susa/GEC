# import argparse
# import glob
# import os
# import json
# import time
# import logging
# import random
# import re
# from itertools import chain
# from string import punctuation
#
# import nltk
# # nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
#
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
#
# from transformers import (
#     AdamW,
#     T5ForConditionalGeneration,
#     T5Tokenizer,
#     get_linear_schedule_with_warmup
# )
#
# import random
# import numpy as np
# import torch
# import datasets
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#
# set_seed(42)
#
# pd.set_option('display.max_colwidth', None)
#
# df = pd.read_csv('C:/Users/bibis/PycharmProjects/GEC/app/data/W-sentence.csv')
# # print(df.shape)
#
# # print(df.head())
#
# from transformers import (
#     T5ForConditionalGeneration, T5Tokenizer,
#     Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
#   )
#
# from torch.utils.data import Dataset, DataLoader
#
# model_name = 't5-base'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
#
#
# def calc_token_len(example):
#     return len(tokenizer(example).input_ids)
#
#
# from sklearn.model_selection import train_test_split
#
# train_df, test_df = train_test_split(df, test_size=0.10, shuffle=True)
#
# test_df['input_token_len'] = test_df['input'].apply(calc_token_len)
#
# from datasets import Dataset
#
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)
#
# from torch.utils.data import Dataset, DataLoader
#
#
# class GrammarDataset(Dataset):
#     def __init__(self, dataset, tokenizer, print_text=False):
#         self.dataset = dataset
#         self.pad_to_max_length = False
#         self.tokenizer = tokenizer
#         self.print_text = print_text
#         self.max_len = 64
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def tokenize_data(self, example):
#         input_, target_ = example['input'], example['output']
#
#         # tokenize inputs
#         tokenized_inputs = tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
#                                      max_length=self.max_len,
#                                      return_attention_mask=True)
#
#         tokenized_targets = tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
#                                       max_length=self.max_len,
#                                       return_attention_mask=True)
#
#         inputs = {"input_ids": tokenized_inputs['input_ids'],
#                   "attention_mask": tokenized_inputs['attention_mask'],
#                   "labels": tokenized_targets['input_ids']
#                   }
#
#         return inputs
#
#     def __getitem__(self, index):
#         inputs = self.tokenize_data(self.dataset[index])
#
#         if self.print_text:
#             for k in inputs.keys():
#                 print(k, len(inputs[k]))
#
#         return inputs
#
#
# dataset = GrammarDataset(test_dataset, tokenizer, True)
#
#
# from datasets import load_metric
# rouge_metric = load_metric("rouge")
#
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')
#
# # defining training related arguments
# batch_size = 16
# args = Seq2SeqTrainingArguments(output_dir="/content/drive/MyDrive/c4_200m/weights",
#                                 evaluation_strategy="steps",
#                                 per_device_train_batch_size=batch_size,
#                                 per_device_eval_batch_size=batch_size,
#                                 learning_rate=2e-5,
#                                 num_train_epochs=1,
#                                 weight_decay=0.01,
#                                 save_total_limit=2,
#                                 predict_with_generate=True,
#                                 fp16=True,
#                                 gradient_accumulation_steps=6,
#                                 eval_steps=500,
#                                 save_steps=500,
#                                 load_best_model_at_end=True,
#                                 logging_dir="/logs",
#                                 report_to="wandb")
#
# import nltk
#
# nltk.download('punkt')
# import numpy as np
#
#
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # Rouge expects a newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
#
#     result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     # Extract a few results
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#
#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
#     return {k: round(v, 4) for k, v in result.items()}
#
#
# trainer = Seq2SeqTrainer(model=model,
#                          args=args,
#                          train_dataset=GrammarDataset(train_dataset, tokenizer),
#                          eval_dataset=GrammarDataset(test_dataset, tokenizer),
#                          tokenizer=tokenizer,
#                          data_collator=data_collator,
#                          compute_metrics=compute_metrics)
#
#
# trainer.train()

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'deep-learning-analytics/GrammarCorrector'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def correct_grammar(input_text, num_return_sequences):
    batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(
        torch_device)
    translated = model.generate(**batch, max_length=64, num_beams=4, num_return_sequences=num_return_sequences,
                                temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


text = 'El nu vreau s-ă vină.'
print(correct_grammar(text, num_return_sequences=2))

