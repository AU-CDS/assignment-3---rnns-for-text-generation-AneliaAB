# %% [markdown]
# RNNs for text generation

# %%
#data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
data_dir = os.path.join("../../news_data/")

# %%
all_comments = []
for filename in os.listdir(data_dir):
    if 'Comments' in filename:
        comments_df = pd.read_csv(data_dir + filename)
        all_comments.extend(list(comments_df["commentBody"].values))

# %%
all_comments = [c for c in all_comments if c != "Unknown"]
len(all_comments)

# %%
#function from in-class notebook
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

# %%
corpus = [clean_text(x) for x in all_comments]
corpus[:10]

# %%
tokenizer = Tokenizer()
## tokenization
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# %%
def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

# %%
inp_sequences, total_words = get_sequence_of_tokens(tokenizer, corpus)
inp_sequences[:10]

# %%
def generate_padded_sequences(input_sequences):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

# %%
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


