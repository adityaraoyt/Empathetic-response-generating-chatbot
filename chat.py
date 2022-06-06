import random
import json
import pymongo
import torch
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from nltk_utils import bag_of_words, tokenize
from pymongo import MongoClient
from Conversations import Log
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
df = pd.read_csv('twocolumndata.csv',encoding='ISO-8859-1')
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

from sklearn.model_selection import train_test_split
X = df.Text
y = df.Emotion
X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

naivebayes = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
naivebayes.fit(X_train, y_train)


bot_name = "Doc"
with open('intents.json', 'r') as f:
    intents = json.load(f)
print("Let's chat! (type 'quit' to exit)")



def get_response(msg):
    # sentence = "do you use credit cards?"
    client = MongoClient("")
    db=client.get_database('chatbot')
    log=Log()
    sessionID="--"
    if msg == "quit":
        return "Goodbye!"
    else:
        tag=naivebayes.predict([msg])
        for intent in intents['intents']:
            if tag == intent["tag"]:
                global res
                res=random.choice(intent['responses'])
                log.saveConversations(sessionID, msg, res, tag[0], db)
                return res