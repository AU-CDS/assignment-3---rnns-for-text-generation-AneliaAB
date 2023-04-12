# %%
# data processing tools
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
#in-class function
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 


# %%
def gather_folder():
    folder = input("write the name of the folder that you would like to use (ex. news_data): ")
    return folder

# %% [markdown]
# Load data and create corpus

# %%
#in-class function
def load_data(filepath):
    data_dir = os.path.join(filepath)

    all_headlines = []
    for filename in os.listdir(data_dir):
        if 'Articles' in filename:
            article_df = pd.read_csv(data_dir + filename)
            all_headlines.extend(list(article_df["headline"].values))
    
    all_headlines = [h for h in all_headlines if h != "Unknown"]

    return all_headlines

def create_corpus(filepath):
    all_headlines = load_data(filepath)
    corpus = [clean_text(x) for x in all_headlines]

    return corpus

# %%
filepath = f"../../{gather_folder()}/"

create_corpus(filepath)

print(create_corpus(filepath)[:10])

# %% [markdown]
# Tokenizing the data

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
def tokenizing(filepath):
    corpus = create_corpus(filepath)
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    inp_sequences = get_sequence_of_tokens(tokenizer, corpus)

    return tokenizer, inp_sequences

# %%
def padded_sequences(filepath):
    tokenizer, input_sequences = tokenizing(filepath)
    total_words = len(tokenizer.word_index) + 1
    corpus = create_corpus(filepath)

    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len, total_words

# %% [markdown]
# Creating model

# %%
def create_model(filepath):
    predictors, label, max_sequence_len, total_words = padded_sequences(filepath)
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 
                        10, 
                        input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

# %% [markdown]
# Saving model

# %%
def save_model(filepath):
    predictors, label, max_sequence_len, total_words = padded_sequences(filepath)
    model = create_model(filepath)
    export_path = "../out/"

    history = model.fit(predictors, 
                    label, 
                    epochs=100,
                    batch_size=128, 
                    verbose=1)
    
    tf.keras.saving.save_model(model, export_path, overwrite=True, save_format=None)

# %%
save_model(filepath)

