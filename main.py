import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
from tensorflow import *
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)
    
words = []
labels  = []
docs = []

#data preprocessing

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
        