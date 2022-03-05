import nltk 
import numpy as np
# Bottom ssl is workaround for broken script on punkt donwloadm which returns a loading ssl error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
#End of error workaround

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#Imports needed from nltk

#Our Tokenizer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Stemming Function
def stem(word):
    return stemmer.stem(word.lower())

#Bag of Words Function
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


#DONE: Test our function with the below sentence to visualize Tokenization. 
#Testing our Tokenizer
test_sentence = "I will not live in peace until I find the Avatar!"
print(tokenize(test_sentence))
#DONE CONT: What is the purpose of tokenizing our text? 
"""
The tokenizer breaks down a sentence into a list of its individual words.
"""

#DONE: Test our Stemming function on the below words. 
words = ["Organize", "organizes", "organizing", "disorganized"]
for word in words:
    print(stem(word))
#DONE CONT: How does stemming affect our data?
"""
Stemming cleans the data by identifying slight variations of 
a given word (past tense, present participle, etc.) as the same word. 
This helps the Bag of Words function accurately count instances of 
the same word and its variations.
"""


#DONE: Implement the above Bag of Words function on the below sentence and words. 
print("Testing our bag_of_words function")
sentence = ["I", "will", "now", "live", "in", "peace", "until", "I", "find", "the", "Avatar"]
words = ["hi", "hello", "I", "you", "the", "bye", "in", "cool", "wild", "find"]
print(bag_of_words(sentence, words))
print("--------------")
#DONE (CONTINUED): What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?
"""
The Bag of Words function translates each sentence into a row in a matrix in which 
each component contains either a 0 or 1 to indicate the presence of the token listed 
for that matrix column. We use Bag of Words here because the chatbot needs to be 
familiar with all of the words in the dataset and their frequencies. TF-IDF is ideal 
for finding words relevant to each document and ignoring "filler" words like "the", "a", 
"of", etc. Word2Vec is a great system for identifying words with similar semantic meaning. 
Thus, TF-IDF and Word2Vec, although typically better-performing than Bag of Words, are not 
required for the purposes of this project.
"""


