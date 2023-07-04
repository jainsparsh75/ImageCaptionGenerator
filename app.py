from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications import EfficientNetB5
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Input,Add
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from tqdm import tqdm


vocab = np.load('vocab (3).npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v: k for k, v in vocab.items()}


print("vocabulary loaded")



# preparing model architecture
embedding_size = 128
max_len = 40
vocab_size = len(vocab)

image_model = keras.Sequential(
    [
        layers.Dense(embedding_size, input_shape=(2048,), activation='relu'),
        layers.RepeatVector(max_len),
    ]
)


language_model = keras.Sequential(
    [
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len),
        layers.Bidirectional(LSTM(256, return_sequences=True)),
        layers.Dense(embedding_size),
    ]
)


conca = Concatenate()([image_model.output, language_model.output])
x = Bidirectional(LSTM(128, return_sequences=True))(conca)
x = Bidirectional(LSTM(512, return_sequences=False))(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


#loading model weights
model.load_weights('mine_model_weights (3).h5')

print("=" * 150)
print("MODEL LOADED")

# loading image model
EfficientNetB5 = EfficientNetB5(include_top=False,weights='imagenet',input_shape=(456,456,3),pooling='avg')


print("=" * 150)
print("EfficientNetB5 MODEL LOADED")

app = Flask(__name__)

# to have always latest file in the static
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, EfficientNetB5, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("=" * 50)
    print("IMAGE SAVED")

    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (456,456))

    image = np.reshape(image, (1,456,456, 3))

    incept = EfficientNetB5.predict(image).reshape(1, 2048)

    print("=" * 50)
    print("Predict Features")

    text_in = ['startofseq']

    final = ''

    print("=" * 50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)

    return render_template('after.html', data=final)


if __name__ == "__main__":
    app.run(debug=True)