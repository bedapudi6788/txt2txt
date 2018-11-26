import os
import pickle

import numpy as np
np.random.seed(6788)

import tensorflow as tf
tf.set_random_seed(6788)

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, SimpleRNN, Activation, dot, concatenate, Bidirectional
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint

# Defining constants here
encoding_vector_size = 128

# Placeholder for max lengths of input and output which are user configruable constants
max_input_length = None
max_output_length = None

char_start_encoding = 1
char_padding_encoding = 0

def build_sequence_encode_decode_dicts(input_data):
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2

def encode_sequences(encoding_dict, sequences, encoding_vector_size):
    encoded_data = np.zeros(shape=(len(sequences), encoding_vector_size))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), encoding_vector_size)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text


def generate(text, input_encoding_dict, model, max_input_length, max_output_length):
    encoder_input = encode_sequences(input_encoding_dict, [text], max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        
        if decoder_input[:,i] == char_padding_encoding:
            return decoder_input[:,1:]

    return decoder_input[:,1:]

def infer(text, model, params):
    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate(text, input_encoding_dict, model, max_input_length, max_output_length)
    return decode_sequence(output_decoding_dict, decoder_output[0])


def build_params(input_data = [], output_data = [], params_path = 'test_params', max_lenghts = (5,5)):
    if os.path.exists(params_path):
        print('Loading the params file')
        params = pickle.load(open(params_path, 'rb'))
        return params
    
    print('Creating params file')
    input_encoding, input_decoding, input_dict_size = build_sequence_encode_decode_dicts(input_data)
    output_encoding, output_decoding, output_dict_size = build_sequence_encode_decode_dicts(output_data)
    params = {}
    params['input_encoding'] = input_encoding
    params['input_decoding'] = input_decoding
    params['input_dict_size'] = input_dict_size
    params['output_encoding'] = output_encoding
    params['output_decoding'] = output_decoding
    params['output_dict_size'] = output_dict_size
    params['max_input_length'] = max_lenghts[0]
    params['max_output_length'] = max_lenghts[1]

    pickle.dump(params, open(params_path, 'wb'))
    return params

def convert_training_data(input_data, output_data, params):
    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    encoded_training_input = encode_sequences(input_encoding, input_data, max_input_length)
    encoded_training_output = encode_sequences(output_encoding, output_data, max_output_length)
    training_encoder_input = encoded_training_input
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
    training_decoder_input[:, 0] = char_start_encoding
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]
    x=[training_encoder_input, training_decoder_input]
    y=[training_decoder_output]
    return x, y

def build_model(params_path = 'test/params'):
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    print('Input encoding', input_encoding)
    print('Input decoding', input_decoding)
    print('Output encoding', output_encoding)
    print('Output decoding', output_decoding)


    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    encoder = Embedding(input_dict_size, 128, input_length=max_input_length, mask_zero=True)(encoder_input)
    encoder = Bidirectional(LSTM(128, return_sequences=True, unroll=True), merge_mode='concat')(encoder)
    encoder_last = encoder[:,-1,:]

    decoder = Embedding(output_dict_size, 256, input_length=max_output_length, mask_zero=True)(decoder_input)
    decoder = LSTM(256, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(128, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model, params



if __name__ == '__main__':
    input_data = ['123', '213', '312', '321', '132', '231']
    output_data = ['123', '123', '123', '123', '123', '123']
    build_params(input_data = input_data, output_data = output_data, params_path = 'test/params', max_lenghts=(10, 10))
    
    model, params = build_model(params_path='test/params')

    input_data, output_data = convert_training_data(input_data, output_data, params)
    
    checkpoint = ModelCheckpoint('test/checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=20, callbacks=callbacks_list)
