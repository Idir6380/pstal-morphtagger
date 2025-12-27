import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_preparation import (
    build_char_vocab, build_morph_vocabs,
    prepare_data, create_dataloader, PAD_ID
)
from model import MorphTagger
from config import *


def compute_loss_and_accuracy(model, dataloader, criterion, trait_names, device):
    """Calcule la loss et l'accuracy sur un dataloader."""
    model.eval()

    total_loss = 0.0
    total_correct = {name: 0 for name in trait_names}
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            in_enc = batch[0].to(device)
            ends = batch[1].to(device)
            # Les sorties sont les elements 2 a N du batch
            targets = {name: batch[i+2].to(device) for i, name in enumerate(trait_names)}

            outputs = model(in_enc, ends)

            # Calculer la loss (somme sur tous les traits)
            batch_loss = 0
            for name in trait_names:
                logits = outputs[name].view(-1, outputs[name].shape[-1])
                target = targets[name].view(-1)
                batch_loss += criterion(logits, target)

            total_loss += batch_loss.item()

            # Calculer l'accuracy par trait (masquer le padding)
            mask = (targets[trait_names[0]] != PAD_ID)
            total_tokens += mask.sum().item()

            for name in trait_names:
                preds = torch.argmax(outputs[name], dim=-1)
                correct = ((preds == targets[name]) * mask).sum().item()
                total_correct[name] += correct

    avg_loss = total_loss / len(dataloader)
    accuracies = {name: (total_correct[name] / total_tokens) * 100 for name in trait_names}
    avg_accuracy = sum(accuracies.values()) / len(accuracies)

    return avg_loss, accuracies, avg_accuracy


def save_model(model, char_vocab, trait_vocabs, path):
    """Sauvegarde le modele et les vocabulaires."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_vocab': dict(char_vocab),
        'trait_vocabs': {name: dict(vocab) for name, vocab in trait_vocabs.items()},
        'hyperparameters': {
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'dropout': DROPOUT,
            'max_c': MAX_C,
            'max_w': MAX_W,
        }
    }, path)
    print(f"Modele sauvegarde: {path}")


def fit(model, train_loader, dev_loader, trait_names, char_vocab, trait_vocabs, device):
    """Entraine le modele."""
    print("=" * 60)
    print("ENTRAINEMENT")
    print("=" * 60)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dev_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            in_enc = batch[0].to(device)
            ends = batch[1].to(device)
            targets = {name: batch[i+2].to(device) for i, name in enumerate(trait_names)}

            optimizer.zero_grad()

            outputs = model(in_enc, ends)

            # Loss = somme des losses sur tous les traits
            loss = 0
            for name in trait_names:
                logits = outputs[name].view(-1, outputs[name].shape[-1])
                target = targets[name].view(-1)
                loss += criterion(logits, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Evaluation sur dev
        dev_loss, dev_accs, dev_avg_acc = compute_loss_and_accuracy(
            model, dev_loader, criterion, trait_names, device
        )

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")
        print(f"  Dev Avg Acc: {dev_avg_acc:.2f}%")

        # Early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            save_model(model, char_vocab, trait_vocabs, MODEL_SAVE_PATH)
            print(f"  -> Meilleur modele sauvegarde!")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/5")

        if patience_counter >= 5:
            print("\nEarly stopping!")
            break

    return model


def main():
    print("=" * 60)
    print("PREPARATION DES DONNEES")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Construire les vocabulaires sur le train
    print("\n1. Construction des vocabulaires...")
    char_vocab = build_char_vocab(TRAIN_FILE)
    trait_vocabs = build_morph_vocabs(TRAIN_FILE)

    # Preparer les donnees train
    print("\n2. Preparation des donnees train...")
    train_in_enc, train_ends, train_morphs = prepare_data(
        TRAIN_FILE, char_vocab, trait_vocabs, MAX_C, MAX_W
    )
    train_loader, trait_names = create_dataloader(
        train_in_enc, train_ends, train_morphs, BATCH_SIZE, shuffle=True
    )
    print(f"   {len(train_in_enc)} phrases, {len(train_loader)} batches")

    # Preparer les donnees dev
    print("\n3. Preparation des donnees dev...")
    dev_in_enc, dev_ends, dev_morphs = prepare_data(
        DEV_FILE, char_vocab, trait_vocabs, MAX_C, MAX_W
    )
    dev_loader, _ = create_dataloader(
        dev_in_enc, dev_ends, dev_morphs, BATCH_SIZE, shuffle=False
    )
    print(f"   {len(dev_in_enc)} phrases, {len(dev_loader)} batches")

    # Creer le modele
    print("\n4. Creation du modele...")
    model = MorphTagger(
        vocab_size=len(char_vocab),
        trait_vocabs=trait_vocabs,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametres: {total_params:,}")

    # Entrainer
    model = fit(model, train_loader, dev_loader, trait_names, char_vocab, trait_vocabs, device)

    print("\n" + "=" * 60)
    print("ENTRAINEMENT TERMINE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
