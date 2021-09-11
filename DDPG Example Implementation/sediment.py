import nltk
from nltk.tokenize import word_tokenize
from itertools import chain
import re

# we cant have one big file full of keys and values so I split it up
from SedimentDict_1 import WordScoreChunk1 as wsc1
from SedimentDict_2 import WordScoreChunk2 as wsc2
from SedimentDict_3 import WordScoreChunk3 as wsc3
from SedimentDict_4 import WordScoreChunk4 as wsc4

class WordScore:
    def __init__(self):
        ws1 = wsc1.DWords
        ws2 = wsc2.Dwords
        ws3 = wsc3.Dwords
        ws4 = wsc4.Dwords
        
        # combine all python dictionaries 
        self.wordScores = dict(chain.from_iterable(d.items() for d in (ws1, ws2, ws3, ws4)))
        
    def score(self, article):
        '''Get sediment score'''
        try:
            words = word_tokenize(article)
            words = re.sub("[^a-zA-Z]", " ", str(words))
        except:
            return 0
        return sum([self.wordScores[word] for word in words if word in self.wordScores])