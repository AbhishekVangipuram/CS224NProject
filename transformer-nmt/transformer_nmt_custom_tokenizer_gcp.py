import torch
from torch import nn
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset
from evaluate import load, combine
from tqdm import tqdm

from transformer import *
from dataset import *

from numba import cuda
device = cuda.get_current_device()
device.reset() 

import sys
sys.path.append('../')
from custom_tokenizers.tokenizerv1 import parse_sentence
import json

# get data 
train = load_dataset('csv', data_files='../data/train.csv')['train']
test = load_dataset('csv', data_files='../data/test.csv')['train']
VAL_CUTOFF = 500 # random choice for val loss, full test set is 3110, take 500 away for validation only. 
val = test[:VAL_CUTOFF]
test = test[VAL_CUTOFF:]

# obolo_tokenizer = PreTrainedTokenizerFast(tokenizer_file='../custom_tokenizers/obolo-bpe-tokenizer.json', padding='left')
english_tokenizer = AutoTokenizer.from_pretrained('gpt2', padding='left')

# we construct this vocab by calculating tokens from train data
# obolo_vocab_sents = [set(parse_sentence(s)) for s in train['Obolo']]
# obolo_vocab = set()
# for sent in obolo_vocab_sents:
#     obolo_vocab.update(sent)
# obolo_vocab = special_tokens + list(obolo_vocab)
# obolo_vocab = {token:idx for idx, token in enumerate(obolo_vocab)}

# Serialize data into file:
# json.dump( obolo_vocab, open( "../custom_tokenizers/custom_obolo_tokenizer_vocab.json", 'w' ) )

# Read data from file:
obolo_vocab = json.load( open( "../custom_tokenizers/custom_obolo_tokenizer_vocab.json" ) )

print(len(obolo_vocab))
print(len(english_tokenizer.vocab))

print(DEVICE)

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = parse_sentence
token_transform[TGT_LANGUAGE] = english_tokenizer

vocab_transform[SRC_LANGUAGE] = obolo_vocab 
vocab_transform[TGT_LANGUAGE] = english_tokenizer.vocab

# change this due to custom tokenizer
# init_text_transform(token_transform)
text_transform[SRC_LANGUAGE] = sequential_transforms((lambda text: [obolo_vocab[token] if token in obolo_vocab else UNK_IDX for token in token_transform[SRC_LANGUAGE](text)]), # input to tokens to ids
                                                     tensor_transform)                                                                                                          # add BOS/EOS and create tensor
                                                                   
text_transform[TGT_LANGUAGE] = sequential_transforms((lambda text: token_transform[TGT_LANGUAGE](text).get('input_ids')),
                                                     tensor_transform)    


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(obolo_vocab)
TGT_VOCAB_SIZE = english_tokenizer.vocab_size
# ORIGINAL TRANSFORMER CONFIGURATION
# EMB_SIZE = 512
# NHEAD = 8
# FFN_HID_DIM = 512
# BATCH_SIZE = 64
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3
EMB_SIZE = 1024
NHEAD = 16
FFN_HID_DIM = 1024
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

train_dataloader = generate_dataloader(train['Obolo'], train['English'], BATCH_SIZE)
val_dataloader = generate_dataloader(val['Obolo'], val['English'], BATCH_SIZE)
test_dataloader = generate_dataloader(test['Obolo'], test['English'], BATCH_SIZE)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # save vram 
        del src, tgt
        torch.cuda.empty_cache()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        # save vram 
        del src, tgt
        torch.cuda.empty_cache()

    return losses / len(list(val_dataloader))

from timeit import default_timer as timer

# NUM_EPOCHS = 10
NUM_EPOCHS = 50 

train_losses = []
val_losses = []
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    if epoch % 10 == 0:
        torch.save(transformer, f'checkpoints/transformer_obolo_to_english_custom_{epoch}_epochs.pt')


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        # print(out)
        # print(english_tokenizer.decode(out))
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        # print(next_word)
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # save vram
        del tgt_mask
        torch.cuda.empty_cache()
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    
    return token_transform[TGT_LANGUAGE].decode(tgt_tokens[1:-1])
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("[CLS]", "").replace("[SEP]", "")


preds = []
refs = []
for idx in tqdm(range(len(test))):
    ob, en = test['Obolo'][idx], test['English'][idx]
    refs.append(en)
    pred = translate(transformer, ob)
    preds.append(pred)

# metrics
chrf = load('chrf')
gleu = load('google_bleu')
rouge = load('rouge') 
bleu = load('bleu')
meteor = load('meteor')
metrics = combine([chrf, bleu, rouge, meteor, gleu])

scores = metrics.compute(predictions=preds, references=refs)

import csv 

with open('scores_custom.csv','w') as f:
    w = csv.writer(f)
    w.writerows(scores.items())
with open('preds_custom.csv','w') as f:
    w = csv.writer(f)
    w.writerows(dict(zip(preds, refs)).items())

import matplotlib.pyplot as plt 
import numpy as np

plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.show()