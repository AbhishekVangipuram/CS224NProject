from transformer import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SRC_LANGUAGE = 'ob'
TGT_LANGUAGE = 'en'

# dataset config for our texts
class TextDataset(Dataset):
    def __init__(self, src_list, tgt_list):
        self.src = src_list
        self.tgt = tgt_list
        self.pairs = list(zip(self.src, self.tgt))
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        return self.pairs[idx]

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}

def init_text_transform(token_transform, vocab_transform):
    text_transform[SRC_LANGUAGE] = sequential_transforms((lambda text: token_transform[SRC_LANGUAGE](text).get('input_ids')), # input to tokens to ids
                                                         tensor_transform)                                                    # add BOS/EOS and create tensor
    text_transform[TGT_LANGUAGE] = sequential_transforms((lambda text: token_transform[TGT_LANGUAGE](text).get('input_ids')),
                                                         tensor_transform)    

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def generate_dataloader(src_sents: list[str], tgt_sents: list[str], batch_size: int) -> DataLoader:
    dataset = TextDataset(src_sents, tgt_sents)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)