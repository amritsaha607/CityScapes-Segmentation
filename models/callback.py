import tensorflow as tf
import os
import wandb

class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_dir):
        super(CustomCallback, self).__init__()
        self.val_x = []
        self.val_y = []
        self.best_loss = float('inf')
        self.ckpt_dir = ckpt_dir
        self.last_ckpt = None

#     def on_train_batch_begin(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Training: begin of batch {}; got log keys: {}".format(batch, keys))

#     def on_train_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

#     def on_test_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         self.val_x.append()
#         print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)
        if logs['val_loss']<self.best_loss:
            if self.last_ckpt is not None:
                os.system('rm {}.*'.format(self.last_ckpt))
            self.best_loss = logs['val_loss']
            self.last_ckpt = os.path.join(self.ckpt_dir, 'e_{}_l_{:4f}.ckpt'.format(epoch, self.best_loss))
            self.model.save_weights(self.last_ckpt)
            print("saved successfully")
            
#         print("Epoch {}, logs : {}".format(epoch, logs))
