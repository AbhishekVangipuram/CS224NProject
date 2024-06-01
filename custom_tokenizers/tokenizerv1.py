# Basic Tokenizing WITH MORPHEME ORDER (Milestone)
import re

digraphs = ['gw', 'kw', 'nw', 'gb', 'kp', 'ch', 'ny', 'n̄']  # although n̄ presumably will not start a reduplicated word
underdot = '̣'
tones = ['̀', '̂']
vowels = 'aeiouọ'
consonants = "fknrmbwjsyltgpdch"

# this function returns root(+suffixes) if can be inferred, -1 otherwise
# if something can be unduplicated, it should be of the form C(digraph possible) + V(underdot) + (tone? skip for now...) + same C, + (i) + same V...
# ignore tones
# asume string input
# THIS SHIT SHOULD NOT BE NAMED STR
def verb_unduplicate(str):
    str = str.replace(r'̀|̂', '')  # remove tones, ASSUMES TONE WOULD ONLY BE ON REDUPLICATED SECTION IF PRESENT. ALSO ERASES THAT INFO. 99% NOT IMPORTANT BUT COULD BE BETTER.
    if (len(str) < 1) or (str[0] not in consonants):
        return -1
    C = str[0]
    if str[:2] in digraphs:
        C = str[:2]
    
    vowel_ind = len(C)
    if (vowel_ind >= len(str)) or (str[vowel_ind] not in vowels):
        return -1
    V = str[vowel_ind]
    # if ((vowel_ind+1) < len(str)) and (str[vowel_ind + 1] == underdot):    # more nasty uggo code
    #     V += underdot
    #     vowel_ind += 1

    if ((len(C+V)+len(C+V)) <= len(str)) and (C+V == str[len(C+V) : len(C+V)+len(C+V)]):
        return str[len(C+V):]
    if ((len(C+V)+len(C+'i'+V)) <= len(str)) and (C+'i'+V == str[len(C+V) : len(C+V)+len(C+'i'+V)]):
        return str[len(C+V):]
    return -1
# tests = ['chechieen̄', 'wuwulu', 'kpọkpọkọ', 'rọriọọn̄', 'sisi', 'cheche', 'rọrọbọ', 'kwa', 'asdf', 'babe', 'biabe', 'babia']
# print([verb_unduplicate(test) for test in tests])
    

#confused about the location and utility of 'ga/ba', not clear from aaron and doesnt seem to occur in this data
# doubled up on neni/keki stuff
# also "kpaba" is a sequence that happens
# prefixes = ['m', 'n', 'n̄', 'i', 'o', 'e', 'ma', 'mo', 'mi', 'me', 'kpa', 'kpo', 'kpe', 'ka', 'si', 'ga', 'ba', 'bo', 'be', 'ni', 'no', 'neni', 'ki', 'ko', 'keki', 'ni', 'no', 'neni']
prefixes = [['m', 'n', 'n̄', 'i', 'o', 'e', 'ma', 'mo', 'mi', 'me'], ['ga', 'ba'], ['ka', 'si'], ['kpa', 'kpo', 'kpe'], ['ba', 'bo', 'be'], ['ni', 'no', 'neni'], ['ki', 'ko', 'keki'], ['ni', 'no', 'neni']]
suffixes = [['ma', 'ni'], ['ge', 'be']]

#ok, so first let's code this to do whatever it can (ignore suffix for now)
#then, add something to print all verb roots this predicts (and the contexts in which they occur)... if they occur with more than 2 prefixes, just take them raw
# so far, no prefixes
# assumes no spaces, punctuation, anything like that yet
raw_verbdict = set()
verbdict = {}
worddict = set()
def shitparse(word):
    original = word
    # return [s]
    parse = []
    # originally did this cyclically until no prefix could be removed. changed to this because prefixes do have fixed order. should help avoid over-identifying verbs/prefixes
    for slot in prefixes:
        for prefix in slot:
            if word.startswith(prefix):
                parse.append(prefix)
                word = word[len(prefix):]
                break
            elif word and word[0] in tones:
                parse.append(word[0])
                word = word[1:]
    #how am i going to approach doing reduplication OVER prefixing?? well, for now, I'm not! !!!!!!!! NOTE NOTE NOTE !!!!!!
    tail = []
    root = ''
    last_prefix = parse[-1] if len(parse) else ''
    first_suffix = ''
    if word:
        if verb_unduplicate(word) != -1:
            parse.append('REDUP')
            word = verb_unduplicate(word)
        
        tail = []
        for slot in suffixes:
            for suffix in slot:
                if word.endswith(suffix):
                    tail = [suffix] + tail
                    word = word[:-len(suffix)]
                    break
        if word:
            root = word
            parse.append(word)
        if len(tail):
            first_suffix = tail[0]

    # CLEAN UP
    # be warned: doing this rn might keep you from finding verb forms like miin̄... remove this if you're going to do the manual verb list thing
        #(which you highkey should do...)
        #let's try appending prefixes or suffixes!
    # weak protection against ridiculous verb forms:
    def is_valid_root(root):
        # check for starts with consonant and has vowel
        if root:
            if root[0] not in consonants:
                return False
            C = root[0:2] if root[0:2] in digraphs else root[0]
            if len(root) == len(C):
                return False
            else:
                V = root[len(C)]
                if V not in vowels:
                    return False
        # check all vowels are identical (sorry writing this very piece by piece)
        # except there might be <i>+vowel in CjV contexts
        vowel_set = set()
        glide_removed = root
        for vowel in vowels:
            glide_removed = glide_removed.replace('i'+vowel, vowel)
        for char in glide_removed:
            if char in vowels:
                vowel_set.add(char)
        if len(vowel_set) != 1:
            return False
        
        # DELETE THIS CHECK LATER:
        if 'REDUP' in root:
            return False
        return True
                
    if len(parse)+len(tail) > 1:   # this stuff is messy and needs cleaning, DW about it until clear ab intentions tho
        if root:
            if not is_valid_root(root):
                if is_valid_root(last_prefix+root):
                    root = last_prefix+root
                    parse.pop()
                    parse[-1] = root
                elif is_valid_root(root+first_suffix):
                    root += first_suffix
                    parse[-1] = root
                    tail.pop(0)
                else:
                    return [original]
        else:
            if is_valid_root(first_suffix):
                root = first_suffix
            elif is_valid_root(last_prefix):
                root = last_prefix
            else:
                return [original]
        if root in verbdict:
            verbdict[root].add(original)
        else:
            verbdict[root] = {original}

    parse += tail
    if len(parse) > 1:
        raw_verbdict.add(original)
    # if len(parse) > 6:
        # print(parse)
    for token in parse:
        worddict.add(token)
    return parse
#print(shitparse('isikiwuwulube'))


def parse_sentence(s):
    sections = s.split(' ')  # how am I gonna deal with lack of spaces in detokenization?
    sections = [section for section in sections if section.strip()]
    # print('sections = ' + str(sections))
    words = []
    for section in sections:
        words += re.split(r'([,\.\?!;:\-“”‘’\[\]\(\)])', section)
    words = [word for word in words if word.strip()]
    # print('words = ' + ' '.join(words))
    parse = []
    for word in words:
        parse += shitparse(word)
    # print('parse = ' + '  '.join(parse))
    return parse

# print(parse_sentence('okumugwem otutumu inyi emi ibe, “gwun̄ ebilene, tap iman̄ me lek'))


# sometimes there is gonna be competition between parses (e.g., o-be-bene... should i be smart and say no, be needs to co-occur with e, or should i just roll with it)
# also like in like isisi the first si could be reduplicative or the prefix as well... this one is genuinely ambiguous
# for now do not take tone as a morpheme, try to be smart about working around it
# remember that the real tryhard thing is to try to decompose morphemes, i.e., to analyze ma as 1SG, FUT

import torch

class CustomOboloTokenizer:
    def __init__(self, string_to_list_tokens, vocab, unk_token_id, pad_token_id):
        self.string_to_list_tokens = string_to_list_tokens  # e.g. parse_sentence
        self.vocab = vocab                                  # e.g. obolo_vocab
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.device = 'cpu'
    
    def to(self, device):
        self.device = device

    def __call__(self, str, return_tensors=None):
        token_list = self.string_to_list_tokens(str)
        id_list = [(self.vocab[token] if token in self.vocab else self.unk_token_id) for token in token_list]
        attention_mask = [1 for _ in range(len(id_list))]
        if return_tensors == "pt":
            id_list = torch.tensor(id_list).to(self.device)
            attention_mask = torch.tensor(attention_mask).to(self.device)
        return {'input_ids': id_list,
                'attention_mask': attention_mask}
        
    def decode(self, ids, skip_special_tokens=True):
        return " ".join([self.vocab[id] for id in ids if id >= (3 if skip_special_tokens else 0)])