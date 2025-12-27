# TP2 - Analyse Morphologique avec Apprentissage Multi-taches

## Objectif

Developpement d'un systeme d'analyse morphologique qui predit les traits morphologiques (TM) des mots francais (Gender, Number, Tense, etc.) a l'aide d'un classifieur neuronal RNN base sur les caracteres.

## Architecture

```
in_enc (caracteres)
       |
   Embedding
       |
      GRU
       |
    gather (fins de mots via 'ends')
       |
   14 x Linear (un par trait)
```

## Structure du projet

```
Tp2/
├── src/
│   ├── data_preparation.py   # Vocabulaires, encodage, DataLoader
│   ├── model.py              # MorphTagger (RNN multi-taches)
│   ├── train_morph.py        # Entrainement
│   ├── predict_morph.py      # Prediction
│   └── config.py             # Hyperparametres
├── model/                    # Modele sauvegarde (.pt)
├── logs/                     # Logs d'entrainement et evaluation
└── notebooks/                # Notebooks d'experimentation
```

## Installation

```bash
cd Tp2
python3 -m venv env
source env/bin/activate
pip install conllu torch tqdm
```

## Utilisation

### Entrainement
```bash
cd src
python train_morph.py
```

### Prediction
```bash
python predict_morph.py <input.conllu> <output.conllu>
```

### Evaluation
```bash
python ../../lib/evaluate.py -g <gold.conllu> -p <pred.conllu> -c feats
```

## Resultats

| Metrique | Score |
|----------|-------|
| Accuracy globale | 92.59% |
| micro-avg F-score | 95.80% |
| macro-avg F-score | 92.62% |

### Detail par trait

| Trait | F-score |
|-------|---------|
| Poss | 100.00% |
| Definite | 98.72% |
| Reflex | 97.62% |
| PronType | 97.49% |
| Number | 97.25% |
| NumType | 96.81% |
| Polarity | 96.39% |
| Person | 94.83% |
| VerbForm | 94.21% |
| Gender | 94.20% |
| Tense | 92.74% |
| Mood | 92.70% |
| Voice | 90.27% |

## Hyperparametres

- MAX_C = 200 (max caracteres)
- MAX_W = 20 (max mots)
- EMBED_DIM = 100
- HIDDEN_DIM = 128
- DROPOUT = 0.3
- BATCH_SIZE = 32
- LEARNING_RATE = 0.001

## Corpus

Sequoia (francais) :
- Train : 2184 phrases
- Dev : 402 phrases
- 14 traits morphologiques
