from tensorflow.keras.regularizers import l2
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



bs = 1000
# wdir = './freshly_trained_nets/'
wdir = './good_trained_nets/'
# 不断修改学习率


def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
# 回调函数 Callbacks 是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。然后，在模型上调用 fit() 函数时，
# 可以将 ModelCheckpoint 传递给训练过程

# 这里用来检测并保存验证集上最好的权重，用于在训练过程中进行断点续训，并选择表现最好的模型
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)


# #make residual tower of convolutional blocks
def make_resnet(pair=2, num_blocks=3, num_filters=32, num_outputs=1, bit=[],isct = False,d1=64, d2=64,  ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(3*len(bit)*2*pair,))
    rs = Reshape((pair,6,len(bit)))(inp)
    perm = Permute((1,3,2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv04 = Conv1D(num_filters, kernel_size=7, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02, conv03, conv04], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    # 5*2=10层

    for i in range(depth):
        conv1 = Conv1D(num_filters*4, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*4, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    # add prediction head
    # 展开，全连接层
    flat1 = Flatten()(shortcut)
    dense0 = dropout(flat1, 0.8)
    dense0 = Dense(512, kernel_regularizer=l2(reg_param))(dense0)

    # dense0 = Dense(512, kernel_regularizer=l2(reg_param))(flat1)

    dense0 = BatchNormalization()(dense0)
    dense0 = Activation('relu')(dense0)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_speck_distinguisher(num_epochs, num_rounds=6, depth=5, pair=16,diff=(0x2, 0),isct=False,bit = np.arange(32)):
    print(bit)
    print("pair = ", pair)
    # create the network
    print("num_rounds = ", num_rounds)
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(pair=pair, depth=depth, reg_param=10**-5,bit=bit,isct=isct)
        net.summary()
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data
    # 生成训练数据make_train_data(n, nr, bit,pair=2,diff=(0x80, 0x0), version=128):
    X, Y = sp.make_train_data(int(10**7/16),nr =num_rounds,bit=bit, pair=pair,diff=diff)
    X_eval, Y_eval = sp.make_train_data(int(10**6/16),nr =num_rounds,bit=bit, pair=pair,diff=diff)
    # set up model checkpoint
    check = make_checkpoint(
        wdir+'best'+str(num_rounds)+'r_diff47_depth'+str(depth)+"_num_epochs"+str(num_epochs)+"_pair"+str(pair)+'.h5')
    # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    #train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    net.save(wdir+'model_'+str(num_rounds)+'r_diff47_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pair"+str(pair)+'.h5')
    dump(h.history, open(wdir+'hist'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pair"+str(pair)+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return(net, h)


if __name__ == "__main__":
    # train_speck_distinguisher(num_epochs=5, num_rounds=6, depth=5, pair=16, diff=(0x80, 0), isct=False, bit=[13,14,15,16,17,18,19,20,21,22,23])
    train_speck_distinguisher(num_epochs=10, num_rounds=6, depth=5, pair=16, diff=(0x8000, 0), isct=False,bit=[2,3,4,5,6,7,8,9,10,11,12,13])
    # train_speck_distinguisher(num_epochs=5, num_rounds=6, depth=5, pair=16, diff=(0x800000, 0), isct=False,bit=[24,25,26,27,28,29,30,31,0,1,2])
    # 训练10轮之后，找不到有用的结果
    # 训练10轮之后，找不到有用的结果
    # 训练9轮的时候，使用0x40仍然能够取得较好的分类效果
    # 要使用信息比特的方法进行优势比特搜索？
    # 有没有更好的方式去搜索优势比特，也就是降低时间复杂度？信息比特的方式去降低时间复杂度
    # 优势比特位大概是基本上是相邻排序的，使用优势比特搜索算法，原来的优势比特搜索算法时间复杂度比较高，依靠小版本的优势比特的结论，优势比特搜索算法的优势位是相邻的，同时也是优势的
    # 查找优势比特的算法：
    # 使用自己的方式训练出来的结果比原来的要好很多，数据从55升到70左右
    # 接下来进行轮数的扩展
    # 轮数的扩展先进行需要搜索前面的差分路径，现在先进行差分路径的搜索,b不用进行差分路径的搜索，只要将密文异或就行。
    # 关于分阶段训练的时候，在第二步生成的时候，使用不同的输入差分，只要有一个是真，就可以判断为真，这样就可以扩展神经网络的轮数
    # 现在训练