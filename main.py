import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy 
import tflearn
import tensorflow
import random
import pickle
import json
from flask import Flask, render_template, request

from tensorflow import keras

app = Flask(__name__)

with open('intents1.json') as file:
    data = json.load(file)

input_layer = keras.layers.Input(shape=(224, 224, 3))
irv2 = keras.applications.Xception(weights='imagenet',include_top=False,input_tensor = input_layer)
global_avg = keras.layers.GlobalAveragePooling2D()(irv2.output)
dense_1 = keras.layers.Dense(1,activation = 'sigmoid')(global_avg)
model = keras.Model(inputs=irv2.inputs,outputs=dense_1)
model.summary()
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f) #saving these variables in the file
except:
    #these blank lists are created as we want to go through the json file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            #tokenize the words, stemming. Root words are formed
            wrds = nltk.word_tokenize(pattern) #returns a list of tokenized words
            words.extend(wrds) #adds to words
            docs_x.append(wrds)
            #gives what intent the tag is a part of
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    #removing all the duplicate elements
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #removing any question marks to not have any meaning to our model, and stemming
    words = sorted(list(set(words))) #set removes the duplicate elements

    labels = sorted(labels) #sorting the labels

    #create training and testing output
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    #neural network does not understand strings, but only numbers.
    #presenting the words into numbers
    for x, doc in enumerate(docs_x):
        #list to keep a check on what words are present
        #stemming the words
        bag = [] #bag of words
        wrds = [stemmer.stem(w.lower()) for w in doc]
        #going through the words and adding the information to bag
        for w in words:
            if w in wrds: #word exsits so add 1 to the list
                bag.append(1)
            else: #word does not exsit so add 0 to the list
                bag.append(0)

        output_row = out_empty[:]
        #where the tag is in our labels, and set value to 1 in output
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    #turning the lists into nparrays to be able to fed into model
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
         pickle.dump((words, labels, training, output), f)

#resetting the graph data, to get rid of previous settings
tensorflow.compat.v1.reset_default_graph()
#defines the input shape for our model
net = tflearn.input_data(shape=[None, len(training[0])])
#8 neurons for the first hidden layer
net = tflearn.fully_connected(net, 8)
#8 neurons for the second hidden layer
net = tflearn.fully_connected(net, 8)
#gets probability for each neuron in the output layer,
#the neuron which has the highest probability is selected
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #
net = tflearn.regression(net)
print(model.summary())
#training the model
model = tflearn.DNN(net) #deep neural network
try:
    x
    model.load("model.tflearn")
except:
#we show the model the data 1000 times, the more times it sees the data, the more accurate it should get
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s) #list of tokenized words
    s_words = [stemmer.stem(word.lower()) for word in s_words] #stemming the words

    for x in s_words:
        for i, w in enumerate(words):
            if w == x: #if current word is equal to our word in the sentence, then add 1 to bag list, generates the bag of words
                bag[i] = 1

    return numpy.array(bag)




@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    results = model.predict([bag_of_words(user_input, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()

























