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
#taking unput from user
def gather_folder():
    folder = input("write the name of the folder that you would like to use (ex. news_data)")
    return folder

# %% [markdown]
# Load data and create corpus

# %%
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