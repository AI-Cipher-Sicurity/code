from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.backend import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, AveragePooling1D, Conv1D, MaxPooling1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.nn import dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# import keras
from pickle import dump
import tensorflow as tf
import speck as sp
import numpy as np
import tensorflow
import gc
# import os


bs = 1000
# wdir = './'
wdir = '../good_trained_nets/'
# 不断修改学习率

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


def first_stage(n,num_rounds=10,pair=4):

    
    test_n = int(n)
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model(wdir+"model_6r_diff47_depth5_num_epochs10_pair16.h5")
        net_json = net.to_json()

        net_first = model_from_json(net_json)
        # net_first.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_first.load_weights(wdir+"model_6r_diff47_depth5_num_epochs10_pair16.h5")
        X, Y = sp.make_train_data(int(n/16), nr=num_rounds - 3, bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x00800480, 0x00802084))
        X_eval, Y_eval = sp.make_train_data(int(n/16), nr=num_rounds - 3, bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x00800480, 0x00802084))
    # X, Y = sp.make_train_data(int(n / 32), nr=num_rounds - 3, isct=False,bit=[49,50,51,52,53,57,58,59,60,61,62], pair=pair,diff=(0x0900010000000000, 0x4108010000000000))
    # X_eval, Y_eval = sp.make_train_data(int(n /  32), nr=num_rounds - 3, isct=False,bit=[49,50,51,52,53,57,58,59,60,61,62], pair=pair,diff=(0x0900010000000000, 0x4108010000000000))
    # X, Y = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [37,38,39,40,41,45,46,47,48,49,50], pair=pair,diff=(0x0010000000000090, 0x8010000000000410))
    # X_eval,Y_eval = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [37,38,39,40,41,45,46,47,48,49,50], pair=pair,diff=(0x0010000000000090, 0x8010000000000410))
    # X, Y = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [23,24,25,26,27,31,32,33,34,35,36,37], pair=pair,diff=(0x0000000000240004, 0x0000000001042004))
    # X_eval,Y_eval = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [23,24,25,26,27,31,32,33,34,35,36,37], pair=pair,diff=(0x0000000000240004, 0x0000000001042004))
    # X, Y = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [8,9,10,11,12,16,17,18,19,20,21,22], pair=pair,diff=(0x0000002400040000, 0x0000010420040000))
    # X_eval,Y_eval = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [8,9,10,11,12,16,17,18,19,20,21,22], pair=pair,diff=(0x0000002400040000, 0x0000010420040000))
    # X, Y = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [0,3,4,7,60], pair=pair,diff=(0x0001200020000000, 0x0008210020000000))
    # X_eval,Y_eval = sp.make_train_data(int(n/32), nr=num_rounds-3,isct=False,bit = [0,3,4,7,60], pair=pair,diff=(0x0001200020000000, 0x0008210020000000))
    # X, Y = sp.make_train_data(n, nr=num_rounds-2, pair=pair,diff=(0x8100,0x8102))
    # X_eval,Y_eval = sp.make_train_data(test_n, nr=num_rounds-2, pair=pair,diff=(0x8100,0x8102))
    check = make_checkpoint(
        wdir+'first_best'+str(num_rounds)+"_pair"+str(pair)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_first.fit(X, Y, epochs=5, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval),callbacks=[lr,check])

    net_first.save("net_first.h5")


def second_stage(n,num_rounds=10, pair=4):


    # n=10**8
    test_n = int(n)
    X, Y = sp.make_train_data(int(n/16), nr=num_rounds,isct=False,bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x8000,0))
    X_eval,Y_eval = sp.make_train_data(int(n/16), nr=num_rounds,isct=False,bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x8000,0))
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model("net_first.h5")
        net_json = net.to_json()

        net_second = model_from_json(net_json)
        net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
        # net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_second.load_weights("net_first.h5")
        
    
    check = make_checkpoint(
        wdir+'second_best'+str(num_rounds)+"r_pair"+str(pair)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_second.fit(X, Y, epochs=5, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_second.save("net_second.h5")


def stage_train(n,num_rounds=10, pair=4):

    
    # n=10**8
    # test_n = int(n/8)

    X, Y = sp.make_train_data(int(n/16), nr=num_rounds,isct=False,bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x8000,0))
    X_eval, Y_eval = sp.make_train_data(int(n/16), nr=num_rounds, isct=False,bit=[10,11,12,13,14,15,16,17,18,19,20,21], pair=pair,diff=(0x8000,0))
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync) 
    batch_size = bs * strategy.num_replicas_in_sync
    with strategy.scope():

        net = load_model("net_second.h5")
        net_json = net.to_json()

        net_third = model_from_json(net_json)
        net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
        # net_third.compile(optimizer='adam', loss='mse', metrics=['acc'])
        net_third.load_weights("net_second.h5")

    check = make_checkpoint(
        wdir+'third_best'+str(num_rounds)+"r_pair"+str(pair)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_third.fit(X, Y, epochs=5, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])

    net_third.save(wdir+"model_"+str(num_rounds)+"r_diff39_depth5_num_epochs5_pair"+str(pair)+".h5")
   


if __name__ == "__main__":
    
    # (0040,0000)->(8000,8000)->(8100,8102)->(8000,840a)->(850a,9520)
    first_stage( n=10**7,num_rounds=7,pair=16)
    second_stage(n=10**7,num_rounds=7,pair=16)
    stage_train( n=10**7,num_rounds=7,pair=16)
    # model = load_model(wdir + "model_7r_diff2_depth5_num_epochs5_pair16.h5")
    # X_eval, Y_eval = sp.make_train_data(10000, nr=7, isct=False, bit=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],pair=16, diff=(0x2, 0))
    # loss, accuracy = model.evaluate(X_eval, Y_eval)
    # print(f"Test Loss: {loss}")
    # print(f"Test Accuracy: {accuracy}")
