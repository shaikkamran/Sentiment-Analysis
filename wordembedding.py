
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:27:29 2017

@author: Ravil

"""




from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import random



# Label the documents to be processed by doc2vec
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
    
    def __iter__(self):
        for source, name in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [name+ '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, name in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [name + '_%s' % item_no]))
        return self.sentences
    
    # Sentences are shuffled to train a more accurate doc2vec model
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled



# The source of corpus for training
sources = {'neg.txt':'NEG', 'pos.txt':'POS'}

# Gensim function to label each sentence associated with it
sentences = LabeledLineSentence(sources)

# Doc2vec model initialization
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

# Build vocabulary out of the sources we provide
model.build_vocab(sentences.to_array())

# Train our Doc2vec model 
for epoch in range(10):
    model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=model.iter)


# Doc2vec model is saved and loaded to avoid re-training of model
model.save('./model.d2v')

model = Doc2Vec.load('./model.d2v')

model = Doc2Vec.load('./model.d2v')
model.most_similar('good')
model.most_similar('bad')
model.similarity('woman','girl')
