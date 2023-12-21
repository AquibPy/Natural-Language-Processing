from gensim.models import Word2Vec
import numpy as np

def avg_word2vec(doc,model):
    vectors = [model.wv[word] for word in doc if word in model.wv.index_to_key]
    return np.mean(vectors,axis=0)


sentences = [
    ["I", "love", "programming"],
    ["Programming", "is", "fascinating", "and", "I", "love", "it"]
]

word2vec_model = Word2Vec(sentences,vector_size=100,window=5,min_count=1)

'''
vector_size:Dimensionality of the word vectors.
min_count : Ignores all words with total frequency lower than this.
sg : {0, 1},Training algorithm: 1 for skip-gram; otherwise CBOW.
'''

document_vector = avg_word2vec(sentences[0], word2vec_model)
print(document_vector)