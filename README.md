# Empathetic-response-generating-chatbot
Here we have designed a chatbot that gives users responses based on their emotions/sentiments. Sentiment analysis is performed using the naive-bayes algorithm, which is trained using the twocolumndata.csv dataset that we made on our own. We classified the user's emotion into six categories: Sad, Happy, Angry, Disgust, Fear and Surprise. The responses are chosen at randon from the intents.json file based on the user's emotion. The user's message, the sentiment, the response, along with the timestamp of the response is stored to a MongoDB collection. The GUI for the chatbot was designed using Tkinter.
