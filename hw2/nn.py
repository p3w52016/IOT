import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))

import pandas as pd
import pickle as pkl
import numpy as np
import json
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, concatenate, Conv1D, MaxPooling1D,\
Embedding, Flatten, Reshape, TimeDistributed, Activation
from keras.models import Model, Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=np.nan)

def parse_data(path, mode='tr'):
    if mode == 'tr':
        x_orig = load_data(path)
        y_orig = x_orig[:,:1]
        #y_dict = {0:45523, 1:46678, 2:56743...}
        #y_label = [[0,0,1,0,0....],[1,0,0,0,0.....]]
        y_label = y_one_hot(y_orig.astype(int))
    elif mode == 'te':
        x_orig = load_data(path)
        y_label = x_orig[:, :1].astype(int)

    return x_orig[:,1:], y_label

def load_data(path):
    with open(path, 'rb') as f:
        return np.load(f)

def y_one_hot(y):

    t=np.sort(np.unique(y))
    y_dict = json.load(open('./data/rev_y_dict.json', 'r'))
    y_t = [int(y_dict[str(int(y[i]))]) for i in  range(len(y))]#if int(y[i]) in y_dict.keys()]

    return to_categorical(y_t, num_classes=2563)#, y_dict

def nn_model():
    loc_in = Input(shape=(8480, 1))

    cnn = Conv1D(100, 3)(loc_in)
    cnn = BatchNormalization()(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Dropout(0.75)(cnn)
    cnn = Conv1D(64, 7)(loc_in)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.75)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv1D(64, 10)(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.75)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling1D(2)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(32, activation='relu')(cnn)
    cnn_out = Dropout(0.5)(cnn)


    other_in = Input(shape=(20,), name='other_input')
    dnn = concatenate([cnn_out, other_in])
    dnn = Dense(64, activation='relu')(dnn)
    dnn = Dropout(0.5)(dnn)
    dnn = BatchNormalization()(dnn)
    output = Dense(2563, activation='softmax', name='output')(dnn)

    model = Model(inputs=[loc_in, other_in], outputs=[output])
    model.summary()

    return model

def embbed_nn():
    loc_in = Input(shape=(40,), name='lstm_input')
    loc_embed = Embedding(input_dim=448, output_dim=64, input_length=40)(loc_in)#N*40*64
    loc_latent = Flatten()(loc_embed)

    #td_loc = TimeDistributed(Dense(64), input_shape=(20, 512))(loc_embed)
    #td_loc = Flatten()(td_loc)

    loc_lstm_in = Reshape((20, -1))(loc_latent)
    loc_lstm = LSTM(512, dropout=0.75)(loc_lstm_in)
    #loc_lstm = LSTM(512, dropout=0.75)(loc_lstm)
    #loc_lstm = Flatten()(loc_lstm)
    loc_lstm = Dense(32, activation='relu')(loc_lstm)

    dist_in = Input(shape=(19, 1,), name='lstm_dist_input')
    dist_lstm = LSTM(10, dropout=0.75)(dist_in)
    dist_lstm = Dense(5, activation='relu')(dist_lstm)

    other_in = Input(shape=(20,), name='other_input')

    dnn = concatenate([loc_latent, dist_lstm,  other_in])
    dnn = Dense(512, activation='relu')(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dropout(0.5)(dnn)
    dnn = Dense(64, activation='relu')(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dropout(0.5)(dnn)
    output = Dense(2563, activation='softmax', name='output')(dnn)

    model = Model(inputs=[loc_in, dist_in, other_in ], outputs=[output])
    model.summary()

    return model

def ts_embbed_nn():
    loc_in = Input(shape=(24, 1,), name='loc_input')

    loc_embed = Embedding(input_dim=423, output_dim=64, input_length=24)(loc_in)#N*24*64
    loc_latent = Flatten()(loc_embed)
    loc_latent = Reshape((-1, 1))(loc_latent)

    loc_in2 = Input(shape=(40, ), name='loc_input2')

    loc_embed2 = Embedding(input_dim=448, output_dim=64, input_length=40)(loc_in2)#N*24*64
    loc_latent2 = Flatten()(loc_embed2)

    loc_lstm = LSTM(128, input_shape=(24, 1), dropout=0.5, name='loc_lstm')(loc_in)
    loc_lstm = Dense(32, activation='relu')(loc_lstm)
    loc_lstm = Dropout(0.5)(loc_lstm)

    dist_in = Input(shape=(19, 1,), name='lstm_dist_input')
    dist_drop = Dropout(0.3)(dist_in)
    dist_lstm = LSTM(64, dropout=0.75, name='dist_lstm')(dist_drop)
    dist_lstm = Dense(32, activation='relu')(dist_lstm)
    dist_lstm = Dropout(0.5)(dist_lstm)

    loc_cnn = Conv1D(64, 64, activation='relu', name='loc_cnn')(loc_latent)
    loc_cnn = Dropout(0.5)(loc_cnn)
    loc_cnn = Conv1D(64, 2, activation='relu')(loc_cnn)
    loc_cnn = Dropout(0.5)(loc_cnn)
    loc_cnn = Conv1D(64, 5, activation='relu')(loc_cnn)
    loc_cnn = Dropout(0.5)(loc_cnn)
    loc_cnn = MaxPooling1D(20)(loc_cnn)
    loc_cnn = Flatten()(loc_cnn)
    loc_cnn = Dense(64, activation='relu')(loc_cnn)
    loc_cnn = Dropout(0.5)(loc_cnn)

    other_in = Input(shape=(20,), name='other_input')

    lstm_in = concatenate([loc_lstm, dist_lstm], name='merge_lstm')
    lstm_in = Dense(32, activation='relu')(lstm_in)
    lstm_in = Dropout(0.5)(lstm_in)

    dnn = concatenate([lstm_in, loc_cnn, loc_latent2, other_in], name='concat')
    dnn = Dense(128, activation='selu')(dnn)
    dnn = Dropout(0.5)(dnn)
    output = Dense(2563, activation='softmax', name='output')(dnn)

    model = Model(inputs=[loc_in2, loc_in, dist_in, other_in], outputs=[output])
    model.summary()

    return model



def train(batch_size=256, epochs=1000, path='./model/422_embed_dnn.h5', aug=False):
    x_train, y_train = parse_data('./data/422_oh/test_validation_data_422_oh.npy')
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0, random_state=4)

    #Add 422 to 422_ts
    x2_train, y2_train = parse_data('./data/422/test_validation_data_422.npy')
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2_train, y2_train, test_size = 0, random_state=4)

    if aug:
        #Add aug validate
        x_aug, y_aug = parse_data('./data/422_ts/test_validation_data_422_ts.npy')
        x_aug, xt_aug, y_aug, yt_aug = train_test_split(x_aug, y_aug, test_size = 0, random_state=4)

        #Add aug 422 to aug 422_ts
        x2_aug, y2_aug = parse_data('./data/422/test_validation_data_422.npy')
        x2_aug, xt2_aug, yt_aug, yt2_aug = train_test_split(x2_aug, y2_aug, test_size = 0, random_state=4)

        #Validate*10=2000*10=20000
        x_aug = np.tile(x_aug, (10, 1))
        y_aug = np.tile(y_aug, (10, 1))
        x2_aug = np.tile(x2_aug, (10, 1))
        print(y_train.shape, y_aug.shape)
        x_train = np.concatenate([x_train, x_aug])
        y_train = np.concatenate([y_train, y_aug])
        x2_train = np.concatenate([x2_train, x2_aug])

    def nn(cb=None):
        model = nn_model()
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x_loc = x_train[:,1:8481].reshape(-1, 8480, 1)
        x_other = np.concatenate([x_train[:, :1], x_train[:, 8481:]], axis=1)
        model.fit([x_loc, x_other], y_train, batch_size=batch_size, epochs=epochs,\
                    validation_split=0, callbacks=[cb])
        model.save('./model/422_oh_1000.h5')

    def embbed():
        x_loc = x_train[:, 1:41]
        x_dist = x_train[:, 41:]
        x_other = np.concatenate([x_train[:, :1], x_dist], axis=1)


        model = embbed_nn()
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([x_loc, x_dist.reshape(-1, 19, 1), x_other], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        model.summary()
        model.save(path)

    def ts_embbed(cb=None):
        x_loc = x_train[:, 1:25]
        x_dist = x_train[:, 25:]
        x_other = np.concatenate([x_train[:, :1], x_dist], axis=1)
        x2_loc = x2_train[:, 1:41]

        model = ts_embbed_nn()
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([x2_loc, x_loc.reshape(-1, 24, 1), x_dist.reshape(-1, 19, 1), x_other], y_train, batch_size=batch_size,\
            epochs=epochs, validation_split=0.1, callbacks=[cb])
        model.save('./model/422_ts_embbed_dnn_mix_test_only.h5')

    earlyStopping=EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto')

    nn(cb=earlyStopping)
    #embbed()
    #ts_embbed(cb=earlyStopping)


def test(path='./model/422_embed_dnn.h5'):
    def train_split():
        x, y = parse_data('./data/422_ts/train_data_422_ts.npy', mode='te')
        yt_dict = json.load(open('./data/y_dict.json', 'r'))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=40)

        model = load_model(path)

        xt_loc = x_test[:, 1:25]
        xt_dist = x_test[:, 25:]
        xt_other = np.concatenate([x_test[:, :1], xt_dist], axis=1)
        pred = model.predict([xt_loc, xt_other])
        pred = np.array([sorted(range(len(pred[i])), key=lambda k: pred[i][k], reverse=True) for i in range(len(pred))])
        pred = pred.reshape(1, -1)    #2563
        pred = np.array([int(yt_dict[str(pred[0][i])]) for i in range(pred.shape[1])]).reshape(-1, 2563)

        #y_test = [np.argmax(y_test[i]) for i in range(len(y_test))]
        #y_test = np.array([yt_dict[str(y_test[i])] for i in range(len(y_test))])

        pred = np.concatenate([y_test.reshape(-1, 1), pred], axis=1)
        with open('./output/422_pred.pkl', 'wb') as f:
            pkl.dump(pred, f)

        #with open('./output/y_180_pred.pkl', 'wb') as f:
        #    pkl.dump(y_test, f)

    def validate():
        x_test, y_test = parse_data('./data/422/test_validation_data_422.npy', 'te')
        yt_dict = json.load(open('./data/y_dict.json', 'r'))

        model = load_model(path)

        xt_loc = x_test[:, 1:41]
        xt_dist = x_test[:, 41:]
        xt_other = np.concatenate([x_test[:, :1], xt_dist], axis=1)
        pred = model.predict([xt_loc, xt_dist.reshape(-1, 19, 1), xt_other])
        pred = np.array([sorted(range(len(pred[i])), key=lambda k: pred[i][k], reverse=True) for i in range(len(pred))])
        pred = pred.reshape(1, -1)    #2563
        pred = np.array([int(yt_dict[str(pred[0][i])]) for i in range(pred.shape[1])]).reshape(-1, 2563)

        #y_test = [np.argmax(y_test[i]) for i in range(len(y_test))]
        #y_test = np.array([yt_dict[str(y_test[i])] for i in range(len(y_test))])

        pred = np.concatenate([y_test.reshape(-1, 1), pred], axis=1)
        print(pred.shape)
        with open('./output/valid_422_pred.pkl', 'wb') as f:
            pkl.dump(pred, f)

    def te():
        x_test = np.load('./data/422_oh/test_data_422_oh.npy')
        x2_test = np.load('./data/422/test_data_422.npy')
        yt_dict = json.load(open('./data/y_dict.json', 'r'))

        model = load_model('./model/422_oh_1000.h5')
        model.summary()
        #exit()

        xt_loc = x_test[:,1:8481].reshape(-1, 8480, 1)
        xt_other = np.concatenate([x_test[:, :1], x_test[:, 8481:]], axis=1)

        #xt_loc = x_test[:, 1:25]
        #xt_dist = x_test[:, 25:]
        #xt_other = np.concatenate([x_test[:, :1], xt_dist], axis=1)
        #xt2_loc = x2_test[:, 1:41]

        pred = model.predict([xt_loc, xt_other])
        pred = np.array([sorted(range(len(pred[i])), key=lambda k: pred[i][k], reverse=True) for i in range(len(pred))])
        pred = pred.reshape(1, -1)    #2563
        pred = np.array([int(yt_dict[str(pred[0][i])]) for i in range(pred.shape[1])]).reshape(-1, 2563)

        with open('./submit/422_oh_1000.pkl', 'wb') as f:
            pkl.dump(pred, f)

    #train_split()
    #validate()
    te()

if __name__ == '__main__':
    #train()
    test()
