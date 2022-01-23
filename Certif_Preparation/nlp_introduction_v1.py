import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import string

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

# ##################

sequences = tokenizer.texts_to_sequences(sentences)
print(sentences)
print(sequences)

# ###################

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

print("######### Test without <OOV> #########")
print(test_data)
test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)

print("######### Test with <OOV> #########")
print(test_data)
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)

print("#### Padding with pre ####")

padded = pad_sequences(sequences)
print(padded)

print("#### Padding with post ####")

padded = pad_sequences(sequences, padding='post')
print(padded)

print("#### Padding with truncating pre ####")

padded = pad_sequences(sequences, padding='post', maxlen=6)
print(padded)

print("#### Padding with truncating post ####")

padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)

print("#### Removing stopwords ####")

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

sentence = "<h1>Welcome everybody</h1><h2> My name is Romerik and I'm a boy who like artificial intelligence and I'm " \
           "working my best to be so good in that</h2> "
print(sentence)
soup = BeautifulSoup(sentence, features="html.parser")
print(soup)
sentence = soup.get_text()
print(sentence)

words = sentence.split()
filtered_sentence = ""
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)

print(sentences)

table = str.maketrans('', '', string.punctuation)
words = sentence.split()
filtered_sentence = ""
for word in words:
    word = word.translate(table)
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "
sentences.append(filtered_sentence)

print(sentences)
