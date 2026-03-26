from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertTokenizer


def whitespace(t):
    return t.split()


def bpe_tokens(t):
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=50)
    tok.train_from_iterator([t], trainer)
    out = tok.encode(t)
    return out.tokens


def wordpiece_tokens(t):
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    return tok.tokenize(t)


print("Enter text:")
t = input()

w = whitespace(t)
b = bpe_tokens(t)
wp = wordpiece_tokens(t)

print("\nWhitespace Tokens:")
print(w)

print("\nBPE Tokens:")
print(b)

print("\nWordPiece Tokens:")
print(wp)