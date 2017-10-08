# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:27:29 2017

@author: Ravil

"""


import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models, similarities
from gensim.models import TfidfModel

#Load corpus and remove stopwords
stop_words = set(stopwords.words('english'))
file1 = open("train-pos.txt")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt','a')
        appendFile.write(" "+r)
        appendFile.close()


with open("train-neg.txt","w+") as file:
    documents = file.readlines()
    lines=[i.split() for i in documents]

    # remove common words and tokenize

    stoplist = set("a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours	 ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves . , / \ ! @ # $ % ^ & ( ) _ + =".split())
    texts = [[file for word in document.lower().split() if word not in stoplist]
         

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]


# To filter out less relevant words using tf-idf model
tfidf = models.TfidfModel(corpus)
corpus_tfidf=tfidf[corpus]
for i in corpus_tfidf:
    print(i)


