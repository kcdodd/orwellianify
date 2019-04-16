import numpy as np
import spacy
from spacy import strings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en")

vecs = nlp.vocab.vectors
print(len(vecs))

not_vec = nlp.vocab[ nlp.vocab.strings["false"] ].vector - nlp.vocab[ nlp.vocab.strings["true"] ].vector

not_vec /= np.linalg.norm(not_vec)

def opposite( word ):
  if word not in nlp.vocab.strings:
    print(f"'{word}' is not in the vocabulary.")
    exit()

  word = nlp.vocab[ nlp.vocab.strings[word] ]
  word_vec = word.vector

  not_word_vec = word_vec + not_vec

  m_norm = np.inf
  i_m = -1

  for i in vecs:

    r = vecs[i] - word_vec
    dist = np.linalg.norm( r )
    diss = np.dot( not_vec, r )

    norm = diss - dist

    if norm < m_norm:
       i_m = i
       m_norm = norm

  return i_m, nlp.vocab.strings[i_m], m_norm


x = opposite("hello")[1] + " " + opposite("world")[1]
print(x)
