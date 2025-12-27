
TRAIN_FILE = "../../sequoia/sequoia-ud.parseme.frsemcor.simple.train"
DEV_FILE = "../../sequoia/sequoia-ud.parseme.frsemcor.simple.dev"
MODEL_SAVE_PATH = "../model/morph_model.pt"

# Donnees
MAX_C = 200      # max caracteres
MAX_W = 20       # max mots

EMBED_DIM = 100
HIDDEN_DIM = 128
DROPOUT = 0.3

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
