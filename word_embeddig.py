import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

voc_size = 10000
sent_length = 8
## 10 feature dimesnions
dim=10

sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good']

one_hot_repr = [one_hot(word,voc_size) for word in sent]
print("\nOne Hot Representation\n",one_hot_repr)

### Padding
embedded_docs = pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)
print("\nPre Padding Representation\n",embedded_docs)

model = Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam','mse')

print(model.predict(embedded_docs[0]))