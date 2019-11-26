import os
import pickle
import logging

import numpy as np
np.random.seed(6788)

import tensorflow as tf
try:
    tf.set_random_seed(6788)
except:
    pass

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, SimpleRNN, Activation, dot, concatenate, Bidirectional, GRU
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint


# Placeholder for max lengths of input and output which are user configruable constants
max_input_length = None
max_output_length = None

char_start_encoding = 1
char_padding_encoding = 0

def build_sequence_encode_decode_dicts(input_data):
    """
    Builds encoding and decoding dictionaries for given list of strings.

    Parameters:

    input_data (list): List of strings.

    Returns:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    decoding_dict (dict): Reverse of above dictionary.

    len(encoding_dict) + 2: +2 because of special start (1) and padding (0) chars we add to each sequence.

    """
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                # Using 2 + because our sequence start encoding is 1 and padding encoding is 0
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2

def encode_sequences(encoding_dict, sequences, max_length):
    """
    Encodes given input strings into numpy arrays based on the encoding dicts supplied.

    Parameters:

    encoding_dict (dict): Dictionary with chars as keys and corresponding integer encodings as values.

    sequences (list): List of sequences (strings) to be encoded.

    max_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    Returns:
    
    encoded_data (numpy array): Reverse of above dictionary.

    """
    encoded_data = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_data[i][j] = encoding_dict[sequences[i][j]]
    return encoded_data


def decode_sequence(decoding_dict, sequence):
    """
    Decodes integers into string based on the decoding dict.

    Parameters:

    decoding_dict (dict): Dictionary with chars as values and corresponding integer encodings as keys.

    Returns:

    sequence (str): Decoded string.

    """
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += decoding_dict[i]
    return text


def generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio):
    """
    Main function for generating encoded output(s) for encoded input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    all_completed_beams (list): List of completed beams.

    """
    if not isinstance(texts, list):
        texts = [texts]

    min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
    min_cut_off_len = min(min_cut_off_len, max_output_length)

    all_completed_beams = {i:[] for i in range(len(texts))}
    all_running_beams = {}
    for i, text in enumerate(texts):
        all_running_beams[i] = [[np.zeros(shape=(len(text), max_output_length)), [1]]]
        all_running_beams[i][0][0][:,0] = char_start_encoding

    
    while len(all_running_beams) != 0:
        for i in all_running_beams:
            all_running_beams[i] = sorted(all_running_beams[i], key=lambda tup:np.prod(tup[1]), reverse=True)
            all_running_beams[i] = all_running_beams[i][:max_beams]
        
        in_out_map = {}
        batch_encoder_input = []
        batch_decoder_input = []
        t_c = 0
        for text_i in all_running_beams:
            if text_i not in in_out_map:
                in_out_map[text_i] = []
            for running_beam in all_running_beams[text_i]:
                in_out_map[text_i].append(t_c)
                t_c+=1
                batch_encoder_input.append(texts[text_i])
                batch_decoder_input.append(running_beam[0][0])


        batch_encoder_input = encode_sequences(input_encoding_dict, batch_encoder_input, max_input_length)
        batch_decoder_input = np.asarray(batch_decoder_input)
        batch_predictions = model.predict([batch_encoder_input, batch_decoder_input])

        t_c = 0
        for text_i, t_cs in in_out_map.items():
            temp_running_beams = []
            for running_beam, probs in all_running_beams[text_i]:
                if len(probs) >= min_cut_off_len:
                    all_completed_beams[text_i].append([running_beam[:,1:], probs])
                else:
                    prediction = batch_predictions[t_c]
                    sorted_args = prediction.argsort()
                    sorted_probs = np.sort(prediction)

                    for i in range(1, beam_size+1):
                        temp_running_beam = np.copy(running_beam)
                        i = -1 * i
                        ith_arg = sorted_args[:, i][len(probs)]
                        ith_prob = sorted_probs[:, i][len(probs)]
                        
                        temp_running_beam[:, len(probs)] = ith_arg
                        temp_running_beams.append([temp_running_beam, probs + [ith_prob]])

                t_c+=1

            all_running_beams[text_i] = [b for b in temp_running_beams]
        
        to_del = []
        for i, v in all_running_beams.items():
            if not v:
                to_del.append(i)
        
        for i in to_del:
            del all_running_beams[i]

    return all_completed_beams

def infer(texts, model, params, beam_size=3, max_beams=3, min_cut_off_len=10, cut_off_ratio=1.5):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    beam_size (int): Beam size at each prediction.

    max_beams (int): Maximum number of beams to be kept in memory.

    min_cut_off_len (int): Used in deciding when to stop decoding.

    cut_off_ratio (float): Used in deciding when to stop decoding.

        # min_cut_off_len = max(min_cut_off_len, cut_off_ratio*len(max(texts, key=len)))
        # min_cut_off_len = min(min_cut_off_len, max_output_length)

    Returns:
    
    outputs (list of dicts): Each dict has the sequence and probability.

    """
    if not isinstance(texts, list):
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    all_decoder_outputs = generate(texts, input_encoding_dict, model, max_input_length, max_output_length, beam_size, max_beams, min_cut_off_len, cut_off_ratio)
    outputs = []

    for i, decoder_outputs in all_decoder_outputs.items():
        outputs.append([])
        for decoder_output, probs in decoder_outputs:
            outputs[-1].append({'sequence': decode_sequence(output_decoding_dict, decoder_output[0]), 'prob': np.prod(probs)})

    return outputs

def generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    input_encoding_dict (dict): Encoding dictionary generated from the input strings.

    model (kerasl model): Loaded keras model.

    max_input_length (int): Max length of input sequences. All sequences will be padded to this length to support batching.

    max_output_length (int): Max output length. Need to know when to stop decoding if no padding character comes.

    Returns:
    
    outputs (list): Generated outputs.

    """
    if not isinstance(texts, list):
        texts = [texts]

    encoder_input = encode_sequences(input_encoding_dict, texts, max_input_length)
    decoder_input = np.zeros(shape=(len(encoder_input), max_output_length))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_output_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
        
        if np.all(decoder_input[:,i] == char_padding_encoding):
            return decoder_input[:,1:]

    return decoder_input[:,1:]

def infer_greedy(texts, model, params):
    """
    Main function for generating output(s) for given input(s)

    Parameters:

    texts (list/str): List of input strings or single input string.

    model (kerasl model): Loaded keras model.

    params (dict): Loaded params generated by build_params

    Returns:
    
    outputs (list): Generated outputs.

    """
    return_string = False
    if not isinstance(texts, list):
        return_string = True
        texts = [texts]

    input_encoding_dict = params['input_encoding']
    output_decoding_dict = params['output_decoding']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']

    decoder_output = generate_greedy(texts, input_encoding_dict, model, max_input_length, max_output_length)
    if return_string:
        return decode_sequence(output_decoding_dict, decoder_output[0])

    return [decode_sequence(output_decoding_dict, i) for i in decoder_output]


def build_params(input_data = [], output_data = [], params_path = 'test_params', max_lenghts = (5,5)):
    """
    Build the params and save them as a pickle. (If not already present.)

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params_path (str): Path for saving the params.

    max_lenghts (tuple): (max_input_length, max_output_length)

    Returns:
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
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
    """
    Encode training data.

    Parameters:

    input_data (list): List of input strings.

    output_data (list): List of output strings.

    params (dict): Generated params (encoding, decoding dicts ..).

    Returns:
    
    x, y: encoded inputs, outputs.

    """
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

def build_model(params_path = 'test/params', enc_lstm_units = 128, unroll = True, use_gru=False, optimizer='adam', display_summary=True):
    """
    Build keras model

    Parameters:

    params_path (str): Path for saving/loading the params.

    enc_lstm_units (int): Positive integer, dimensionality of the output space.

    unroll (bool): Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

    use_gru (bool): GRU will be used instead of LSTM

    optimizer (str): optimizer to be used

    display_summary (bool): Set to true for verbose information.


    Returns:

    model (keras model): built model object.
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    # generateing the encoding, decoding dicts
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)


    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(encoder_input)
    # using concat merge mode since in my experiments it gave the best results same with unroll
    if not use_gru:
        encoder = Bidirectional(LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = concatenate([forward_h, backward_h])
        encoder_c = concatenate([forward_c, backward_c])
    
    else:
        encoder = Bidirectional(GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)        
        encoder_outs, forward_h, backward_h= encoder
        encoder_h = concatenate([forward_h, backward_h])
    

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)
    
    decoder = Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length, mask_zero=True)(decoder_input)

    if not use_gru:
        decoder = LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=[encoder_h, encoder_c])
    else:
        decoder = GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=encoder_h)


    # luong attention
    attention = dot([decoder, encoder_outs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(enc_lstm_units, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        model.summary()
    
    return model, params



if __name__ == '__main__':
    input_data = ['123', '213', '312', '321', '132', '231']
    output_data = ['123', '123', '123', '123', '123', '123']
    build_params(input_data = input_data, output_data = output_data, params_path = 'params', max_lenghts=(10, 10))
    
    model, params = build_model(params_path='params')

    input_data, output_data = convert_training_data(input_data, output_data, params)
    
    checkpoint = ModelCheckpoint('checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=40, callbacks=callbacks_list)
