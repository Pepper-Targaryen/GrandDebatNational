# -*- coding: utf-8 -*-

"""
It takes one or two hours to finish the clustering with all the data,
if you want to test the code,
just take like the first 3000 or so ones.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import string
import stop_words
from sklearn.cluster import KMeans
import pandas as pd
import random

"""
To use the two packages of nltk, do 
nltk.download("stopwords")
and
nltk.download('punkt')
in the python console
after you've installed nltk
if you don't want certain word that appears but not filtered by me,
add it in the list ['', '’', '``', '\'\'', '»', '...','«', 'nan', '--']
in the 3rd line of the get_stop_words() function
"""

def get_stop_words():
    custom_stop_words = set(stopwords.words('french') +
                            list(string.punctuation) +
                            ['', '’', '``', '\'\'', '»', '...','«', 'nan', '--'] +
                            stop_words.get_stop_words('fr'))
    return custom_stop_words


def tokenize(text):
    stemmer = FrenchStemmer()
    words_temp = word_tokenize(text, language='french')
    words_no_prefix = [f[2:] if f.startswith(("l\'","d\'","j\'","n\'","c\'")) else f for f in words_temp]
    words_no_prefix = [f[3:] if f.startswith(("qu\'")) else f for f in words_no_prefix]
    words_prefect = [stemmer.stem(word) for word in words_no_prefix if not word.isdigit()]
    return words_prefect

df = pd.read_csv("data/DEMOCRATIE_ET_CITOYENNETE.csv", low_memory=False)

#Use this of you only want to do one certain question
'''
answers = df["QUXVlc3Rpb246MTA3 - En qui faites-vous le plus confiance pour vous faire représenter dans la société et pourquoi ?"]
answers = answers.str.lower()
answers = answers.values.tolist()
answers = [x for x in answers if type(x) is str]
'''

#Use this if you want to combine the responses of all the questions
answers = df.iloc[:,11:]
answers = answers.astype(str)
answers = answers.apply(" ".join, axis =1)
answers = answers.str.lower()
answers = answers.values.tolist()
answers = random.sample(answers, k=3000)


#Vectorize our text
custom_stop_words = get_stop_words()
vectorizer = TfidfVectorizer(stop_words=custom_stop_words,
                            tokenizer=tokenize,
                            max_features=100)
'''
The X in the next line is the matrix transformed from all the text data,
if you wants to split the data to train/text or just divide,
split the X
'''

X = vectorizer.fit_transform(answers)
words = vectorizer.get_feature_names()

#train the cluster
kmeans = KMeans(n_clusters=8, n_init=20)
kmeans.fit(X)

#Display the clustering results
common_words = kmeans.cluster_centers_.argsort()[:, -1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

'''
From now on, it is the code for STEP 2
'''

'''
STEP 2 with Decision tree
'''
from sklearn import tree
y = kmeans.labels_

clf_DT = tree.DecisionTreeClassifier()
clf_DT = clf_DT.fit(X, y)

'''
STEP 2 with Naive Bayes
'''
from sklearn.naive_bayes import MultinomialNB

clf_NB = MultinomialNB()
clf_NB.fit(X, y)

'''
STEP 2 with KNN
'''
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors = 8)
neigh.fit(X, y)

'''
STEP 2 train report
'''
from sklearn import metrics

y_predicted = neigh.predict(X)
print(metrics.classification_report(y, y_predicted))

#TODO
#Use the STEP1 cluster and the STEP2 cluster to predict the same
#block of data, use the result of the cluster as the correct one
#track the performance. Or you can change the parameters as you wish

#TODO IMPORTANT
#observe the clustering result and try to find a meaningful representation
#of each cluster like "people with a negative view", "people who don't believe
# democracy", etc. Like this our report will be more meaningful.