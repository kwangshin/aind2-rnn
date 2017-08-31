import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # We need to creat {P(total length of the series) - T (windows size)} input/output pairs.
    number_of_return_data = len(series) - window_size
    for i in range(number_of_return_data):  
        # The input will be the list with length of window size.
        one_input_data = []
        
        # Define the valud to use outside of for loop.
        j = 0
        # Create the input with the length of windows size from current position.
        for j in range(i, i + window_size):
            one_input_data.append(series[j])
        
        # The output will be only one value at the next position in serise.
        one_output_data = series[j + 1]
        
        # Append the input/output pair.
        X.append(one_input_data)
        y.append(one_output_data)
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # as the first layer in a Sequential model
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # now model.output_shape == (None, 5)
    model.add(Dense(1))
    # now model.output_shape == (None, 1)
    # model.summary()

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # ASCII lowercase - The lowercase letters 'abcdefghijklmnopqrstuvwxyz'.
    ascii_lowercase = string.ascii_lowercase

    # Create the list of allowed characters.
    allowed_characters = list(ascii_lowercase) + punctuation

    # We can consider the space as allowed character.
    allowed_characters.append(' ')

    # Change the not-allowed characters into space.
    for one_character in text:
        if one_character not in allowed_characters:
            text = text.replace(one_character, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    start_position = window_size
    end_position = len(text)
    
    # To calculate the next position.
    counter = 0

    for i in range(start_position, end_position, step_size):  
        # Because the start_position is started from window_size,
        # get the window size string before the current position (i).
        one_input_data = text[counter*step_size:i]
        
        # The output will be only one value at the current position (i).
        one_output_data = text[i]
        
        # Append the input/output pair.
        inputs.append(one_input_data)
        outputs.append(one_output_data)

        counter += 1

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # as the first layer in a Sequential model
    model = Sequential()
    # layer 1 should be an LSTM module with 200 hidden units
    # note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    # layer 3 should be a softmax activation (since we are solving a multiclass classification)
    model.add(Dense(num_chars, activation='softmax'))

    return model