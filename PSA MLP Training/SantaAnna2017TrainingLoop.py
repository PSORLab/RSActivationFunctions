# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:21:21 2020

@author: wilhe

Requires nightly version of tensor flow
"""



# Print Start message to console
print("start")

# Import general scientific computing packages into enviroment.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Import preprocess from sklearn.preprocessing for MinMaxScaler capability.
# Import Tensorflow and Keras frontends.
# Import Layers from Keras to get utilities which can be used to modify layers
# such as setting new activation functions.
# Import EarlyStopping function.
import sklearn.preprocessing as preprocess
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# CURRENTLY NOT USED: Needed to setup multiple CPU hardware acceleration.
#config = tf.ConfigProto(device_count={"CPU": 8})
#tf.keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Get Pandas DataFrame object from file. An select a fraction as the sample.
# Currently, all data is used.
path = "C:\\Users\\wilhe\\Dropbox\\Apps\\Overleaf\\CACE 2020, Relaxations of Activation Functions\Background Papers\\SantaAnna2017Supp\\1-s2.0-S0098135417302053-mmc1revised.csv"
dataset = pd.read_csv(path, names = ["Pa", "Pm", "Pi", "Tad", "Tdj", "Tdr", "Tco", "Q", "T", "L", "D", "yN2", "N2pur", "N2rec", "CH4pur", "CH4rec"])
dataset = dataset.sample(frac = 1)
dataset['split'] = np.random.randn(dataset.shape[0], 1)

test_set_ratio = 0.15
msk = np.random.rand(len(dataset)) <= (1.0 - test_set_ratio)
train_set = dataset[msk]
test_set = dataset[~msk]

# Create feature vector x and target vector y with assigned column names
x = pd.DataFrame(train_set, columns=["Pa", "Pm", "Pi", "Tad", "Tdj", "Tdr", "Tco", "Q", "T", "L", "D", "yN2"])
y = pd.DataFrame(train_set, columns=["N2pur", "N2rec", "CH4pur", "CH4rec"])

# Create minmax scaler and transform the data.
scaler = preprocess.MinMaxScaler()
x[["Pa", "Pm", "Pi", "Tad", "Tdj", "Tdr", "Tco", "Q", "T", "L", "D", "yN2"]] = scaler.fit_transform(x)

# Number of features
p = len(x)

# Storage for results of each training case
fit_results = []

"""
atf contains a list of standard keras activation functions. atf_custom contains
a list of activation functions in the nn module of tensorflow but not in keras
"""
atf = ['tanh']
atf_custom = [[tf.nn.gelu, 'gelu'], [tf.nn.silu, 'silu']]

# Structure of the neural network
ndepth = 1 # number of layers

repeats = 10       # Number of times to train a neural network per configuration
step_size = 1      # Number of neurons per layer to step by in for loop

# Metaparameters for optimizer and earlystopping
val_set_ratio = 0.15                             # Fraction of original dataset to use for tests
split_r = val_set_ratio/(1.0 - test_set_ratio)   # Fraction of x,y to use for validation (15% of original dataset)
mdelta = 0.000001
epoch_lim = 200
verb_set = 1
opt_style = tf.keras.optimizers.Adam(learning_rate = 1E-2)           # optimizer

for n in range(1, 21, step_size):
    print("Neuron Number "+str(n))               # Print outer loop number 
    for r in range(1, repeats+1):
        print("Neuron Number "+str(r))           # Print inner loop number 
        for af in atf:
            print("Neuron Type "+str(af))        # Print neuron type (keras activation functions)
            
            # Create basic sequential model
            model = tf.keras.Sequential()
            
            # Add input layer
            model.add(layers.Input(shape = (p,)))
            
            # Add n depth dense layers of neurons
            for q in range(0, ndepth):
                model.add(layers.Dense(n, af))
            
            # Add a dense output layer of neurons
            model.add(layers.Dense(4))
            
            # Compile the model with specified optimizer and MSE loss function
            model.compile(optimizer=opt_style, loss='mse')
            
            # Specify early stopping settings
            es = EarlyStopping(monitor='val_loss', mode='min', verbose = verb_set, min_delta = mdelta)
            
            # Fit model
            fr = model.fit(x, y, epochs = epoch_lim, validation_split = split_r, callbacks = [es], verbose = verb_set)
            
            # Store results
            fit_results.append([af, n, r, fr.history['loss'][-1], fr.history['val_loss'][-1], 
                                fr.history['loss'], fr.history['val_loss'], model.get_weights()])
        
        for af in atf_custom:
            print("Neuron Type "+str(af))        # Print neuron type (tf.nn activation functions)
            
            # Create basic sequential model
            model = tf.keras.Sequential()
            
             # Add input layer
            model.add(layers.Input(shape = (p,)))
            
            # Add n depth dense layers of neurons
            for q in range(0, ndepth):
                model.add(layers.Dense(n, af[0]))
                
            # Add a dense output layer of neurons
            model.add(layers.Dense(4))
            
            # Compile the model with specified optimizer and MSE loss function
            model.compile(optimizer=opt_style, loss='mse')
            
             # Specify early stopping settings
            es = EarlyStopping(monitor='val_loss', mode='min', verbose = verb_set, min_delta = mdelta)
            
            # Fit model
            fr = model.fit(x, y, epochs = epoch_lim, validation_split = split_r, callbacks = [es], verbose = verb_set)
            
             # Store results
            fit_results.append([af[1], n, r, fr.history['loss'][-1], fr.history['val_loss'][-1],
                                fr.history['loss'], fr.history['val_loss'], model.get_weights()])



# create plot of data frames
active_out = []
repeats_out = []
loss_out = []
val_loss_out = []

# unpack fit results to vectors
for l in fit_results:
    active_out.append(l[0])
    repeats_out.append(l[1])
    loss_out.append(l[3])
    val_loss_out.append(l[4])

# create dictionary used to construct dataframe    
data_dict = {
    'neuron_type' : active_out,
    'n' : repeats_out,
    'loss' : loss_out,
    'val_loss' : val_loss_out,
    }

# make dataframe summarizing results
summary_df = pd.DataFrame(data_dict)


# Construct plot
plot_kind = 'line'

full_labels = ['sigmoid', 'tanh', 'selu', 'elu', 'gelu', 'silu']
full_colors = ['Blue', 'Red', 'DarkOrange', 'Yellow', 'Purple', 'DarkGreen']

for i in range(0,6):
    if i == 0:
        act_frame = summary_df[summary_df['neuron_type'] == full_labels[i]]
        min_frame = act_frame[act_frame.groupby(['neuron_type', 'n'])['loss'].transform(min) == act_frame['loss']]
        last_ax = min_frame.plot(x = 'n', y = ['loss'], kind=plot_kind, color=full_colors[i])
        
act_frame = summary_df[summary_df['neuron_type'] == full_labels[i]]
min_frame = act_frame[act_frame.groupby(['neuron_type', 'n'])['loss'].transform(min) == act_frame['loss']]
last_ax = min_frame.plot(x = 'n', y = ['loss'], kind=plot_kind, ax = last_ax, color=full_colors[i])

print("finished")
