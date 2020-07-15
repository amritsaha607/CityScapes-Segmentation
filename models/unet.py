import tensorflow as tf
import keras

from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.regularizers import l2

from .CustomModel import CustomModel


def conv_factory(x, concat_ax, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''
        BN + ReLU + Conv2D + Dropout(optional)
    '''
    x = BatchNormalization(
        axis=concat_ax,
        gamma_regularizer=l2(weight_decay),
        beta_regularizer=l2(weight_decay),
    )(x)
    x = Activation('relu')(x)
    x = Conv2D(
        nb_filter, (5, 5), dilation_rate=(2, 2),
        kernel_initializer='he_uniform', padding='same',
        kernel_regularizer=l2(weight_decay),
    )(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def denseblock(x, concat_ax, nb_layers, growth_rate, dropout_rate=None, weight_decay=1e-4):
    '''
        nb_layers of conv_factory layers merged together
    '''
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_ax, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
    x = Concatenate(axis=concat_ax)(list_feat)
    return x

def UNet(hp=None):

    '''
        hp : hyperparams (dict)
            lr: learning rate
            # weight_decay : weight_decay
            b1 : beta1
            b2 : beta2
            eps : epsilon
            amsgrad : amsgrad
            opt_name : name
    '''
    if hp is None:
        optimizer = 'adam'
    if hp is not None:
        lr = hp['lr']
        if hp['name'].lower()=='adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, 
                beta_1=hp['b1'], 
                beta_2=hp['b2'], 
                epsilon=hp['eps'], 
                amsgrad=hp['amsgrad'],
                name=hp['opt_name'],
            )

    dr = 0.5
    nr = 2
    mod_inputs = Input((256, 256, 3))

    conv1 = Conv2D(64/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mod_inputs)
    db1 = denseblock(conv1, concat_ax=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    pool1 = MaxPooling2D((2, 2))(db1)

    conv2 = Conv2D(128/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    db2 = denseblock(conv2, concat_ax=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    pool2 = MaxPooling2D((2, 2))(db2)

    conv3 = Conv2D(256/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    db3 = denseblock(conv3, concat_ax=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    pool3 = MaxPooling2D((2, 2))(db3)

    conv4 = Conv2D(512/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    db4 = denseblock(conv4, concat_ax=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    pool4 = MaxPooling2D((2, 2))(db4)

    conv5 = Conv2D(1024/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    db5 = denseblock(conv5, concat_ax=3, nb_layers=3, growth_rate=16, dropout_rate=dr)
    upsampled_db5 = UpSampling2D((2, 2))(db5)
    up5 = Conv2D(512/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upsampled_db5)
    merge5 = Concatenate(axis=3)([BatchNormalization()(db4), BatchNormalization()(up5)])

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    db6 = denseblock(x=conv6, concat_ax=3, nb_layers=3, growth_rate=16, dropout_rate=dr)
    up6 = Conv2D(256/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db6))
    merge6 = Concatenate(axis=3)([BatchNormalization()(db3), BatchNormalization()(up6)])

    conv7 = Conv2D(256/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    db7 = denseblock(x=conv7, concat_ax=3, nb_layers=2, growth_rate=16, dropout_rate=dr)
    up7 = Conv2D(128/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db7))
    merge7 = Concatenate(axis=3)([BatchNormalization()(db2), BatchNormalization()(up7)])

    conv8 = Conv2D(128/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    db8 = denseblock(x=conv8, concat_ax=3, nb_layers=2, growth_rate=16, dropout_rate=dr)
    up8 = Conv2D(64/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db8))
    merge8 = Concatenate(axis=3)([BatchNormalization()(db1), BatchNormalization()(up8)])

    conv9 = Conv2D(64/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    db9 = denseblock(x=conv9, concat_ax=3, nb_layers=2, growth_rate=16, dropout_rate=dr)

    conv10 = Conv2D(32/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9) # final node layer

    conv11 = Conv2D(3, 1, activation='sigmoid')(conv10)

    model = CustomModel(inputs=mod_inputs, outputs=conv11)
    model.compile(
        optimizer=optimizer, 
        loss='MSE', 
        metrics=['acc',],
    )

    return model

