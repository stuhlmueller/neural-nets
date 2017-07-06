# Based on https://github.com/facebookresearch/InferSent/blob/master/encoder/play.ipynb

import os
import torch

GLOVE_PATH = '/infersent/dataset/GloVe/glove.840B.300d.txt'

# torch.load only works if models.py is in the working directory, so:
os.chdir("/infersent/encoder/")

if torch.cuda.is_available():
    print("Using CUDA")
    model = torch.load('infersent.allnli.pickle')
else:
    print("CUDA is not available")
    model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
    model.use_cuda = False

model.set_glove_path(GLOVE_PATH)
model.build_vocab_k_words(K=100000)

sentences = []
with open('/infersent/dataset/MultiNLI/s2.dev.matched') as f:
    for line in f:
        sentences.append(line.strip())

print(len(sentences))
print(sentences[:5])

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))

print(np.linalg.norm(model.encode(['the cat eats.'])))

print(cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0]))