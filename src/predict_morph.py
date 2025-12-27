import sys
import torch
from conllu import parse_incr

from data_preparation import encode_sent_chars, PAD_ID
from model import MorphTagger
from config import MODEL_SAVE_PATH


def load_model(path):
    """Charge le modele et les vocabulaires."""
    checkpoint = torch.load(path, weights_only=False)

    char_vocab = checkpoint['char_vocab']
    trait_vocabs = checkpoint['trait_vocabs']
    hyperparams = checkpoint['hyperparameters']

    model = MorphTagger(
        vocab_size=len(char_vocab),
        trait_vocabs=trait_vocabs,
        embed_dim=hyperparams['embed_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        dropout=hyperparams['dropout']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Creer les vocabulaires inverses pour decoder les predictions
    rev_trait_vocabs = {
        name: {idx: val for val, idx in vocab.items()}
        for name, vocab in trait_vocabs.items()
    }

    return model, char_vocab, trait_vocabs, rev_trait_vocabs, hyperparams


def predict_sentence(model, sent, char_vocab, rev_trait_vocabs, device):
    """Predit les traits morphologiques pour une phrase."""
    # Encoder la phrase
    in_enc, ends = encode_sent_chars(sent, char_vocab)

    # Convertir en tenseurs (batch de 1)
    in_enc_t = torch.LongTensor([in_enc]).to(device)
    ends_t = torch.LongTensor([ends]).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(in_enc_t, ends_t)

    # Decoder les predictions pour chaque mot
    predictions = []
    for i in range(len(sent)):
        word_feats = {}
        for name, output in outputs.items():
            pred_idx = torch.argmax(output[0, i]).item()
            pred_val = rev_trait_vocabs[name].get(pred_idx, "<N/A>")

            # Ne pas ajouter <pad> ou <N/A>
            if pred_val not in ["<pad>", "<N/A>"]:
                word_feats[name] = pred_val

        predictions.append(word_feats if word_feats else None)

    return predictions


def predict_corpus(input_file, output_file, model, char_vocab, rev_trait_vocabs, device):
    """Predit sur tout un corpus et sauvegarde le resultat."""
    num_sentences = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for sent in parse_incr(f_in):
            num_sentences += 1

            # Predire
            predictions = predict_sentence(model, sent, char_vocab, rev_trait_vocabs, device)

            # Mettre a jour les feats
            for token, pred_feats in zip(sent, predictions):
                token['feats'] = pred_feats

            # Ecrire la phrase
            f_out.write(sent.serialize())

    print(f"Phrases traitees: {num_sentences}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_morph.py <input_file> <output_file>")
        print("\nExemple:")
        print("  python predict_morph.py ../../sequoia/sequoia-ud.parseme.frsemcor.simple.dev predicted.conllu")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("=" * 60)
    print("PREDICTION DES TRAITS MORPHOLOGIQUES")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Charger le modele
    print(f"\nChargement du modele: {MODEL_SAVE_PATH}")
    model, char_vocab, trait_vocabs, rev_trait_vocabs, _ = load_model(MODEL_SAVE_PATH)
    model = model.to(device)
    print(f"Traits: {list(trait_vocabs.keys())}")

    # Predire
    print(f"\nPrediction sur: {input_file}")
    predict_corpus(input_file, output_file, model, char_vocab, rev_trait_vocabs, device)

    print(f"\nResultat sauvegarde: {output_file}")
    print("\n" + "=" * 60)
    print("TERMINE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
