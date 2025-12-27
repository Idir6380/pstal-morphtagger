import torch
import torch.nn as nn

class MorphTagger(nn.Module):
    def __init__(self, vocab_size, trait_vocabs, embed_dim=100, hidden_dim=100, dropout=0.3):
        super(MorphTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.trait_names = list(trait_vocabs.keys())

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bias=False,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        self.classifiers = nn.ModuleDict({
            name: nn.Linear(hidden_dim, len(vocab)) for name, vocab in trait_vocabs.items()
        })
    def forward(self, in_enc, ends):
        x = self.embed(in_enc)

        rnn_out, _ = self.gru(x)

        batch_size, max_w = ends.shape
        ends_exp = ends.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        word_repr = rnn_out.gather(1, ends_exp)

        word_repr = self.dropout(word_repr)

        outputs = {}
        for name in self.trait_names:
            outputs[name] = self.classifiers[name](word_repr)

        return outputs
