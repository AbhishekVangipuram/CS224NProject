# %% [markdown]
# ### Important Note
# For this task, we do English -> Obolo (since T5 models exclusively perform English to [other language]) translation; therefore, this has the possibility of harnessing pretrained weights.

# %% [markdown]
# ## Preprocess Data

# %%
model_checkpoint = 't5-small'

# %% [markdown]
# ### Load Data

# %%
# load data
from datasets import load_dataset

train = load_dataset('csv', data_files='../data/train.csv')['train']
test = load_dataset('csv', data_files='../data/test.csv')['train']
VAL_CUTOFF = 500 # random choice for val loss, full test set is 3110, take 500 away for validation only. 
test, val = test.train_test_split(test_size=VAL_CUTOFF).values()

# %% [markdown]
# ### Create Tokenizers

# %%
import json
import sys
sys.path.append('../')
from custom_tokenizers.tokenizerv1 import parse_sentence, CustomOboloTokenizer
from transformers import AutoTokenizer

# t5-small tokenizer, based on sentencepiece. this is what we use for the English portion
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# we can either use the above tokenizer, or a custom tokenizer for the Obolo portion. below, we code our custom Obolo tokenizer
obolo_vocab = json.load(open( "../custom_tokenizers/custom_obolo_tokenizer_vocab.json"))

# modify the obolo vocabulary to match the special tokens of the t5-small tokenizer
modified_obolo_vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2}
for token in obolo_vocab:
    if obolo_vocab[token] <= 4: continue
    modified_obolo_vocab[token] = obolo_vocab[token] - 2

obolo_tokenizer = CustomOboloTokenizer(parse_sentence, modified_obolo_vocab, tokenizer.unk_token_id, tokenizer.pad_token_id)
# %% [markdown]
# For now, we simply use the original sentencepiece tokenizer + new words, not our custom Obolo tokenizer

# %%
# print(f"Number of tokens before: {len(tokenizer)}")
# tokenizer.add_tokens(list(modified_obolo_vocab.keys())[3:])
# print(f"Number of tokens after adding Obolo specific tokens: {len(tokenizer)}")

# %% [markdown]
# ### Preprocess Dataset
# Use the desired tokenizer (in this case, sentencepiece + new vocab) to preprocess dataset from strings to input ids.

# %%
prefix = ""
source_lang = "Obolo"
target_lang = "English"
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples[source_lang]]
    targets = [ex + " </s>" for ex in examples[target_lang]]
    model_inputs = obolo_tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train = train.map(preprocess_function, batched=True)
val = val.map(preprocess_function, batched=True)
test = test.map(preprocess_function, batched=True)

# %% [markdown]
# ## Train

# %%
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# create model and data collator
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# resize the embeddings
# model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForSeq2Seq(obolo_tokenizer, model=model)

# %%
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}-CustomTokenizer",
    eval_strategy = "steps",
    eval_steps = 500,
    save_strategy = 'steps',
    save_steps = 500,
    logging_dir='./logs', 
    logging_steps=50, 
    
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=False
)

# %%
import numpy as np
from evaluate import load, combine

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

chrf = load('chrf')
gleu = load('google_bleu')
rouge = load('rouge') 
bleu = load('bleu')
# meteor = load('meteor')
metrics = combine([chrf, bleu, rouge, gleu])

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metrics.compute(predictions=decoded_preds, references=decoded_labels)
    precisions = result['precisions']
    del result['precisions']
    for i in range(len(precisions)):
        result[f'precisions{i+1}'] = precisions[i]

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# %%
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    data_collator=data_collator,
    tokenizer=obolo_tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %%



