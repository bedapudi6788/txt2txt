# txt2txt - An extremely easy to use seq2seq implementation with Attention for text to text use cases



[![DOI](https://zenodo.org/badge/159134969.svg)](https://zenodo.org/badge/latestdoi/159134969)

# Examples
1. [Adding two numbers](https://colab.research.google.com/drive/11lVvfa2EGYQ0y3O5gA--01iR0J6IRMCk)
2. [More Complex Math and fit_generator](https://colab.research.google.com/drive/1JqBxRiTZ0D1rB3bsw46FaA1McTqrDGCe)


# Installation

```
pip install txt2txt
```

# Training a model
```
from txt2txt import build_params, build_model, convert_training_data

input_data = ['123', '213', '312', '321', '132', '231']
output_data = ['123', '123', '123', '123', '123', '123']

build_params(input_data = input_data, output_data = output_data, params_path = 'test/params', max_lenghts=(10, 10))
    
model, params = build_model(params_path='test/params')

input_data, output_data = convert_training_data(input_data, output_data, params)
    
checkpoint = ModelCheckpoint('test/checkpoint', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(input_data, output_data, validation_data=(input_data, output_data), batch_size=2, epochs=20, callbacks=callbacks_list)
```


# Loading a trained model and running inference
```
from txt2txt import build_model, infer
model, params = build_model(params_path='test/params')
model.load_weights('path_to_checkpoint_file')
infer(input_text, model, params)
```

Note: Checkout https://github.com/bedapudi6788/deepcorrect for pre-trained models for english punctuation correction and grammar correction.

# Requirements
This module needs Keras and Tensorflow. (tested with tf>=1.8.0, keras>=2.2.0).

Tensorflow is not included in setup.py and needs to be installed seperately.

# What's the use of this module

Working with seq2seq tasks in NLP, I realised there aren't any easy to use, simple to understand and good performing libraries available for this. Though libraries like FairSeq or transformer are available they are in general either too complex for a newbie to understand or most probably overkill (and are very tough to train) for simple projects.

This module provides pre-built seq2seq model with Attention that performs excellently on most of the "simple" NLP taks. (Tested with Punctuation correction, transliteration and spell correction)

# To Do

Make number of encoder and decoder layers configurable

Give option to add language model probability in beam search

# License

Although txt2txt is licensed under GPL, if you want to use it commercially without open sourcing your code please email me or raise a issue in this repo so that I can provide you explicit written permission to use as you wish. The only reason for doing this is, it would be nice to know if some company is using my work.
