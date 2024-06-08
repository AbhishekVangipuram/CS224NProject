# Detokenizing
digraphs = ['gw', 'kw', 'nw', 'gb', 'kp', 'ch', 'ny', 'n̄']  # although n̄ presumably will not start a reduplicated word
underdot = '̣'
tones = ['̀', '̂']
vowels = 'aeiouọ'
consonants = "fknrmbwjsyltgpdch"

# with open('nv_tokens.txt', 'r') as file:
#     nv_tokens_txt = file.readlines()
# nv_tokens = set()
# for line in nv_tokens_txt:
#     nv_tokens.add(line.strip())
# with open('ManualVerbList.txt', 'r') as file:
#     canonical_verbs_txt = file.readlines()
# canonical_verbs = set()
# for line in canonical_verbs_txt:
#     canonical_verbs.add(line.strip())
# with open('affixlist.txt', 'r') as file:
#     affixlist_txt = file.readlines()
# affixlist = set()
# for line in affixlist_txt:
#     affixlist.add(line.strip())

# all_tokens = set()
# all_tokens = all_tokens.union(nv_tokens)
# all_tokens = all_tokens.union(canonical_verbs)
# all_tokens = all_tokens.union(affixlist)

def is_prefix(token):
    return len(token) > 1 and token[-1] == '-'
def is_suffix(token):
    return len(token) > 1 and token[0] == '-'

def detokenize(parse):
    result = ''
    prefix_mode = True
    redup_mode = False
    for token in parse:
        if is_prefix(token):
            if token == 'REDUP-':
                redup_mode = True
            elif prefix_mode:
                result += token[:-1]
            else:
                result += ' ' + token[:-1]
            prefix_mode = True
        elif is_suffix(token):
            result += token[1:]
            prefix_mode = False
        else:
            if redup_mode and verb_reduplicate(token) != -1:
                token = verb_reduplicate(token)
                redup_mode = False
            if prefix_mode:
                result += token
            else:
                result += ' ' + token
            prefix_mode = False
    return result

# print(detokenize(['n-', 'ki-', 'REDUP-', 'tap', '-be', 'test']))

def verb_reduplicate(str):
    #input checking and cons/vowel finding
    if (len(str) < 2) or (str[0] not in consonants):
        return -1
    dup = str[0]
    if str[:2] in digraphs:
        dup = str[:2]

    vowel_ind = len(dup)
    if (vowel_ind > len(str)) or (str[vowel_ind] not in vowels):
        return -1

    # this prob easier with regex or something but it's easy enough as is
    # fixing up vowel in case of leading glide or underdot
    if vowel_ind+1 < len(str):
        if str[vowel_ind+1] in vowels:
            vowel_ind += 1
    dup += str[vowel_ind] #lil ugly code here

    return dup+str
# print(verb_reduplicate('tap'))