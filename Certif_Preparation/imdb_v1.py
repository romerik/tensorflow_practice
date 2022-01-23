# coding:utf-8

from bs4 import BeautifulSoup
import string
import tensorflow_datasets as tfds
import tensorflow as tf

print("Program starting...")

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

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("_", " ,_ ")
    sentence = sentence.replace("/", " / ")
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
            imdb_sentences.append(filtered_sentence)


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2500)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
print("Program terminated...")
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# Decoding the first sentence
reverse_word_index = dict(
    [(value, key) for (key, value) in tokenizer.word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])
print(decoded_review)

