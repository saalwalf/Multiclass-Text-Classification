import re
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from indoNLP.preprocessing import replace_slang, replace_word_elongation

# Definisikan fungsi preprocessing
def preprocess(data, text_column):
    # Mengubah teks menjadi huruf kecil, menghilangkan spasi berlebih, dan menghapus baris baru
    data[text_column] = data[text_column].apply(lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))
    
    # Normalisasi ejaan dan menghapus elongasi
    data[text_column] = data[text_column].apply(replace_slang)
    data[text_column] = data[text_column].apply(replace_word_elongation)
    
    # Menghapus pola tertentu seperti retweet, mentions, URLs, dan hashtags
    data[text_column] = data[text_column].apply(lambda x: re.sub(r're \S+', '', x))
    data[text_column] = data[text_column].apply(lambda x: re.sub(r'@\S+', '', x))
    data[text_column] = data[text_column].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
    data[text_column] = data[text_column].apply(lambda x: re.sub(r'#\S+', '', x))
    
    # Menghapus karakter non-alphabet dan menghapus karakter non-ASCII
    data[text_column] = data[text_column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    data[text_column] = data[text_column].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
    
    # Menambahkan stopwords dan kata-kata umum yang tidak diinginkan
    stop = set(stopwords.words('indonesian'))
    kata_umum = {'rt', 'anies', 'baswedan', 'ganjar', 'pranowo', 'prof', 'mahfud', 'md', 
                 'prabowo', 'gibran', 'subianto', 'abah', 'amin', 'yg', 'dkk', 'utk', 
                 'ga', 'nggak', 'gua', 'dgn', 'udah', 'dl', 'dr', 'gw', 'terima kasih',
                 'ya', 'doang', 'nya', 'lu', 'videotron', 'gm', 'sih', 'gak', 'cak imin', 
                 'aja', 'lupa'}
    stop.update(kata_umum)

    # Stemmer untuk bahasa Indonesia
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Fungsi untuk membersihkan dan memproses teks
    def process_text(text):
        # Menghapus stop words
        words = [word for word in text.split() if word not in stop]
        
        # Stemming
        words = [stemmer.stem(word) for word in words]
        
        # Gabungkan kembali kata-kata menjadi string
        return " ".join(words)

    data[text_column] = data[text_column].apply(process_text)

    return data
