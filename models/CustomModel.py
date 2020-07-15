import tensorflow as tf
from tensorflow.keras import Model
from metrics.metrics import acc
import wandb

class CustomModel(Model):

	def train_step(self, data):
		x, y = data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		trainable_vars = self.trainable_variables
		grads = tape.gradient(loss, trainable_vars)
		self.optimizer.apply_gradients(zip(grads, trainable_vars))

		self.compiled_metrics.update_state(y, y_pred)
		logg = {m.name: m.result() for m in self.metrics}

		# self.tot_loss += m.result()['loss']

		return logg

