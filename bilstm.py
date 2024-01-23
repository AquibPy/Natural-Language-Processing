import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


#HyperParameter
voc_size = 5000
sent_length=20
embedding_vector_features = 40

df = pd.read_csv(r"Fake news Classifier\train.csv")
df.dropna(axis=0,inplace=True)

X = df.drop('label',axis=1)
Y = df['label']

## data Preprocessing
messages = X.copy()
messages.reset_index(inplace=True)
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

## OneHot Representation
onehot_repr = [one_hot(words,voc_size)for words in corpus]

## Embedding Representation
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

## Creating Model
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length = sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X_final=np.array(embedded_docs)
Y_final=np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_final, test_size=0.33, random_state=42)

## Model Training
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=32)