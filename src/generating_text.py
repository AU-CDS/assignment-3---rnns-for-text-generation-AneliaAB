# %% [markdown]
# Loading saved model

# %%
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import padded_sequences, tokenizing, gather_folder

filepath = f"../../{gather_folder()}/"

predictors, label, max_sequence_len, total_words = padded_sequences(filepath)
tokenizer, inp_sequences = tokenizing(filepath)

# %%
filepath = "../out/"

# %%
model = tf.keras.saving.load_model(
    filepath, custom_objects=None, compile=True, safe_mode=True)

# %%
def generate_text(seed_text, next_words, model, max_sequence_len):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                    maxlen=max_sequence_len-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list),
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

# %%
print(generate_text(input("write a word to generate new text: "), 5, model, max_sequence_len))


