
import pandas as pd
import numpy as np
from tensorflow import keras

filename = 'restauranttrain.bio.txt'
reviews = []
words = []
sen = []
with open(filename, 'r') as f:
    for line in f:
        if not line.startswith('\n'):
            lines = line.split()
            words.append(lines[1])
            reviews.append(lines)
        if line.startswith('\n'):
            sen.append(words)
            words = []
            continue


entities = []
tags = []
labels = []
for i in reviews:
    if '-' in i[0]:
        store = i[0].split('-')
        tags.append(store[0])
        labels.append(store[1])
        entities.append(i[1])
    else:
        tags.append(i[0])
        labels.append('No Label')
        entities.append(i[1])

list_sen = []
num = 1
for ii in sen:
    list_sen.append('Sentence'+str(num))
    num+=1
    for i_i in range(len(ii)-1):
        list_sen.append('NaN')

data = {'Sentence #':list_sen, 'Word':entities, 'POS':labels, 'Tag':tags}
df = pd.DataFrame(data)
print(df)

from itertools import chain


def get_dict_map(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))

    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

from tensorflow.keras.utils import to_categorical
token2idx, idx2token = get_dict_map(df, 'token')
tag2idx, idx2tag = get_dict_map(df, 'tag')

df['Word_idx'] = df['Word'].map(token2idx)
df['Tag_idx'] = df['Tag'].map(tag2idx)
df = pd.get_dummies(df, columns=['POS'])

val = []
for iii in range(len(df['Word_idx'])):
    val.append(int(float(df['Word_idx'][iii])))

df['Word_idx'] = pd.DataFrame(val)


x_data = np.array(df.iloc[:,3:])
y_data = np.vstack(np.array(df['Tag_idx']))
y_data = to_categorical(y_data, 3)
input = keras.preprocessing.sequence.pad_sequences(x_data, maxlen=11, dtype='int32')
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers


input_dim = len(df['Word_idx'])
n_tags = len(tag2idx)

embedding_layer = Embedding(input_dim=input_dim, output_dim=32, input_length=1000)
inputs = keras.Input(shape=(None,), dtype="int32")
embedded_sequences = embedding_layer(inputs)
x = embedded_sequences
x = layers.LSTM(150, activation='relu', kernel_initializer='he_uniform', dtype='float32')(x)
outputs = layers.Dense(n_tags, activation='relu', kernel_initializer='he_uniform')(x)
model = keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_data, y_data, batch_size=32, epochs=1, validation_split=0.2)

acc = history.history['accuracy']
loss = history.history['loss']

y_predictions = model.predict()

with open('history.txt', 'w') as f:
    for ids in range(len(loss)):
        f.write('acc:' + str(acc[ids]) + 'loss:' + str(loss[ids]) + '\n')



