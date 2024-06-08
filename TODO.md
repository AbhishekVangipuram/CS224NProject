## Modeling
- Retrain Transformer model with tweaks to architecture (more encoder/decoder layers, attention heads, train for more epochs)
- ~~Equip and run Transformer model with new tokenizer~~
- ~~Recreate Obolo BPE tokenizer with updates data~~
- Recreate baselines with updated data
- Can also run an English to Obolo test with BPE because both have tokenization and detokenization
- ~~Run Transformer model for more epochs perhaps~~
- Find how to use mBART50, T5, or LLM such as mixtral or llama3 with a custom tokenizer (not just custom vocab)

## Tokenizers
- Work on flexible tokenizer, such as using plug-and-play morpheme data (such as a morpheme list or other morphological databses)
- Work on detokenization for Obolo
- Work on detokenization with flexible tokenizer




- compare sentences with surgery bpe and normal bpe
- conside reaosns as to why bpe is good, maybe because larger vocab size, maybe can find relations better
- motivate use of chrF++
- consider bpe tokenizer with restrricted vocab size
- consider that bpe has good compressions or something (huffman, information theory?)
