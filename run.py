import torch
from torch import nn
import numpy as np
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset
from evaluate import load, combine
from transformer import *


# get data 
dataset = load_dataset("csv", data_files="data/v3.csv")
data_splits = dataset['train'].train_test_split(0.1)
train, test = data_splits['train'], data_splits['test']

# metrics for later
chrf = load('chrf')
gleu = load('google_bleu')
rouge = load('rouge') 
bleu = load('bleu')
meteor = load('meteor')
metrics = combine([chrf, bleu, rouge, meteor, gleu])


obolo_tokenizer = PreTrainedTokenizerFast(tokenizer_file='data/obolo-bpe-tokenizer.json')
english_tokenizer = AutoTokenizer.from_pretrained('gpt2')

print(obolo_tokenizer.vocab_size)
print(english_tokenizer.vocab_size)
# now this BPE tokenizer is also equipped with a decoder, so we should be able to do Obolo -> English and English -> Obolo

torch.manual_seed(0)

SRC_VOCAB_SIZE = obolo_tokenizer.vocab_size
TGT_VOCAB_SIZE = english_tokenizer.vocab_size
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
