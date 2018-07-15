from nltk.corpus import stopwords
import re
import os
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer

cachedStopWords = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()

stop_words = ['yo', 'so', 'well', 'um', 'a', 'the', 'that', 'you', 'i', 'an', 'is', 'of', 'are',
              'such', 'i', 'me', 'my', 'we', 'it', 'its', 'am', 'is', 'be', "or", 'as',
              'at', 'by', 'to', 'from', 'up', 'in', 'on'
              ]

os.chdir("../../data")
with open(os.path.abspath(os.curdir)+'/stopwords.txt', 'r') as f:
    custom_stop_words = set(f.read().split("\n"))


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in cachedStopWords])


def remove_unwanted_chars(text):
    return re.sub(r'[?|$|.|!|(|)|@|#|=|^|:|;|\|]', r'', text)


def convert_to_lower_case(text):
    return text.lower()


def remove_custom_stop_words(text):
    for w in custom_stop_words:
        pattern = r'\b' + w + r'\b'
        text = re.sub(pattern, '', text)
    return text


def do_pre_processing(text):
    text = remove_non_ascii(text)
    text = remove_unwanted_chars(text)
    return text


def lemmatization(text):
    lemmtized_words = []
    for word in text.split():
        lemmtized_words.append(wordnet_lemmatizer.lemmatize(word))

    return " ".join(lemmtized_words)


def remove_non_ascii(text):
    return unidecode(text)


if __name__ == '__main__':
    print(do_pre_processing("In what year was the Chilean National Museum of Fine Arts built?"))
    # print(remove_unwanted_chars("how much for the $#@%^&*maple syrup? ($20.99)? That's ricidulous!!!"))
