import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
infile = "data_for_trackLength_training.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
np.random.shuffle(Dataset)  #shuffling the data sample to avoid any bias in the training
#print(Dataset)
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1) #labels = TrueTrackLengthInWater
#print(rest)
#print(features[:,2202])
#print(features[:,2201])
#print(labels)
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:2000]
train_y = labels[:2000]

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=2203, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=2, verbose=0)

# checkpoint
filepath="weights_bets.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=10, batch_size=1, callbacks=callbacks_list, verbose=0)
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,10.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
#plt.savefig("Keras/train_test.pdf")
#n, bins, patches = plt.hist(lambdamax, 50, density=1, facecolor='r', alpha=0.75)
#plt.savefig("TrueTrackLengthLambdamaxhist.pdf")
#plt.savefig("TrueTrackLengthhist.pdf")
#plt.scatter(lambdamax,labels)
#plt.savefig("ScatterplotTrueTrackLengthwithlambdamax.pdf")