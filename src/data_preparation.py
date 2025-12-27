from conllu import  parse_incr
from collections import defaultdict
import torch
from  torch.utils.data import TensorDataset, DataLoader

PAD_ID = 0
UNK_ID = 1
ESP_ID = 2


def build_char_vocab(corpus_file):
    char_vocab = defaultdict(lambda : len(char_vocab))

    char_vocab['<pad>'] = PAD_ID
    char_vocab['<unk>'] = UNK_ID
    char_vocab['<esp>'] = ESP_ID

    with open(corpus_file, encoding='utf-8') as f:
        for sent in parse_incr(f):
            for token in sent:
                word = token['form']
                for char in word:
                    _ = char_vocab[char]

    #desactivation de default dict pour le sauvvgarder
    char_vocab.default_factory = None
    print(f'Vocabulaire: {len(char_vocab)} car')
    return char_vocab

def build_morph_vocabs(corps_file):

    trait_vocabs = {}

    with open(corps_file, encoding='utf-8') as f:
        for sent in parse_incr(f):
            for token in sent:
                feats = token['feats']
                if feats:
                    for trait_name, trait_value in feats.items():
                        if trait_name not in trait_vocabs:
                            trait_vocabs[trait_name] = defaultdict(lambda : len(trait_vocabs[trait_name]))
                            trait_vocabs[trait_name]['<pad>'] = 0
                            trait_vocabs[trait_name]['<N/A>'] = 1

                        _ = trait_vocabs[trait_name][trait_value]

    for trait_name in trait_vocabs:
        trait_vocabs[trait_name].default_factory = None

    print(f'traits morphologique: {len(trait_vocabs)}')
    for name, vocab in trait_vocabs.items():
        print(f'{name} : {len(vocab)} values')
    return trait_vocabs

def encode_sent_chars(sent, char_vocab):

    in_enc = [PAD_ID]
    ends =  []

    for i, token in enumerate(sent):
        word = token['form']

        if i > 0:
            in_enc.append(ESP_ID)

        for char in word:
            char_id = char_vocab.get(char, UNK_ID)
            in_enc.append(char_id)
        ends.append(len(in_enc)-1)

    return in_enc, ends

def encode_sent_morphs(sent, trait_vocabs):

    NA_ID = 1

    encoded = {trait_name : [] for trait_name in trait_vocabs}

    for token in sent:
        feats = token['feats']

        for trait_name, vocab in trait_vocabs.items():
            if feats and trait_name in feats:
                value = feats[trait_name]
                encoded[trait_name].append(vocab.get(value, NA_ID))
            else:
                encoded[trait_name].append(NA_ID)

    return encoded


def pad_or_crop(sequence, max_len, pad_value=0):
    """Pad ou crop une séquence à max_len."""
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))


def prepare_data(corpus_file, char_vocab, trait_vocabs, max_c=200, max_w=20):
    """
    Encode tout un corpus et retourne les listes encodées.

    Returns:
        all_in_enc: liste de listes (caractères)
        all_ends: liste de listes (positions fin de mots)
        all_morphs: dict {trait_name: liste de listes}
    """
    all_in_enc = []
    all_ends = []
    all_morphs = {name: [] for name in trait_vocabs}

    with open(corpus_file, encoding='utf-8') as f:
        for sent in parse_incr(f):
            # Encoder entrées
            in_enc, ends = encode_sent_chars(sent, char_vocab)

            # Encoder sorties
            morphs = encode_sent_morphs(sent, trait_vocabs)

            # Pad/crop
            in_enc = pad_or_crop(in_enc, max_c, PAD_ID)
            ends = pad_or_crop(ends, max_w, 0)

            all_in_enc.append(in_enc)
            all_ends.append(ends)

            for name in trait_vocabs:
                morphs[name] = pad_or_crop(morphs[name], max_w, PAD_ID)
                all_morphs[name].append(morphs[name])

    return all_in_enc, all_ends, all_morphs


def create_dataloader(all_in_enc, all_ends, all_morphs, batch_size=32, shuffle=True):
    """
    Crée un DataLoader avec 2 entrées et N sorties.
    """
    # Convertir en tenseurs
    X_chars = torch.LongTensor(all_in_enc)
    X_ends = torch.LongTensor(all_ends)

    # Sorties : un tenseur par trait
    Y_tensors = []
    trait_names = list(all_morphs.keys())
    for name in trait_names:
        Y_tensors.append(torch.LongTensor(all_morphs[name]))

    # Créer le dataset et dataloader
    dataset = TensorDataset(X_chars, X_ends, *Y_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, trait_names



