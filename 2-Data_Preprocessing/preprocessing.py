import pandas as pd
import bz2
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

data = bz2.BZ2File('test.ft.txt.bz2')
data = data.readlines()

sample_size = 200000
data = [x.decode('utf-8') for x in data[:sample_size]]

print(data[0])
df = pd.DataFrame({'text':data})
df['target'] = df.text.apply(lambda x: 1 if '__label__2' in x.split() else 0)
df['text'] = df.text.apply(lambda x: re.sub(r'__label__\d','',x).strip())
df['text'] = df.text.apply(lambda x: re.sub(r'([^ ]+(?<=\.[a-z]{3}))','',x))

print(df.value_counts('target'))

STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(raw_text):
    global STOP_WORDS
    word_list=[]
    for word in raw_text.lower().strip().split():
        word = re.sub(r'\d','',word)
        word = re.sub(r'[^\w\s]', '', word)
        if word not in STOP_WORDS and word!='':
            word_list.append(word)
    return ' '.join(word_list)

df['cleaned_text'] = df.text.apply(lambda x:preprocess_text(x))

df.to_csv("train_data.csv",index=False)