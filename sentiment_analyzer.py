#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Run following commented section once to install if using as .ipynb
# Ignore Otherwise
# import sys
# !{sys.executable} -m pip install tensorflow
# !{sys.executable} -m pip install tensorflow_hub
# !{sys.executable} -m pip install tensorflow_datasets

import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub # These imports will not work unless you've already installed them locally. Colab has them ready.
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)


# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


# In[6]:


# Tokenize all text data

tokenizer_train = []
train_label = []
for idx, (sent, label) in enumerate(train_data):
    sent = sent.numpy()
    new_sent = [word for word in sent.lower().split()]
    tokenizer_train.append(new_sent)
    train_label.append([label.numpy()])

tokenizer_valid = []
valid_label = []
for idx, (sent, label) in enumerate(validation_data):
    sent = sent.numpy()
    new_sent = [word for word in sent.lower().split()]  
    tokenizer_valid.append(new_sent)
    valid_label.append([label.numpy()])
    
tokenizer_test = []
test_label = []
for idx, (sent, label) in enumerate(test_data):
    sent = sent.numpy()
    new_sent = [word for word in sent.lower().split()]
    tokenizer_test.append(new_sent)
    test_label.append([label.numpy()])


# In[7]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenizer_train)
word_to_ix = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(tokenizer_train)
max_len = max([len(x) for x in train_sequences])
train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=max_len)

valid_sequences = tokenizer.texts_to_sequences(tokenizer_valid)
valid_padded = pad_sequences(valid_sequences, padding='post', truncating='post', maxlen=max_len)

test_sequences = tokenizer.texts_to_sequences(tokenizer_test)
test_padded = pad_sequences(test_sequences, padding='post', truncating='post', maxlen=max_len)


# In[8]:


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Embedding(len(word_to_ix) + 1, 64, input_length=max_len),
                                    tf.keras.layers.GlobalAveragePooling1D(),                                    
                                    tf.keras.layers.Dense(10, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[10]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[11]:


train_label = np.array(train_label)
valid_label = np.array(valid_label)
test_label = np.array(test_label)

model.fit(train_padded, train_label, epochs=8, validation_data=(valid_padded, valid_label))


# In[12]:


model.evaluate(test_padded, test_label, verbose=2)


# In[ ]:




