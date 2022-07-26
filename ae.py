import tensorflow as tf
import numpy as np
import os, re
from sklearn.model_selection import train_test_split
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

checkpoint_directory = "training_checkpoints_ae"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")


class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = tf.keras.layers.Dense(384, activation=tf.nn.tanh)
        self.decoder0 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.decoder1 = tf.keras.layers.Dense(2591, activation=tf.nn.sigmoid)

    def call(self,x):
        latent = self.encoder(x)
        x_hat = tf.concat((self.decoder0(latent),self.decoder1(latent)),axis=-1)
        loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x[:,5:-8],x_hat[:,5:-8]),axis=-1) - \
               tf.math.log(tf.reduce_sum(x_hat[:,:5]*x[:,:5],axis=-1)) + tf.reduce_sum((x_hat[:,-8:] - x[:,-8:])**2,axis=-1)
        loss = tf.reduce_mean(loss)
        # loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(x,x_hat),axis=-1)
        # loss = tf.reduce_mean(loss)
        return loss


class AdamWeightDecay(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 **kwargs):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                              epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
                use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        grads, tvars = list(zip(*grads_and_vars))
        if experimental_aggregate_gradients:
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config_optimizer = super(AdamWeightDecay, self).get_config()
        config_optimizer.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config_optimizer

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


def main():
    data = np.load('covid_vumc.npy').astype('float32')
    for i in range(1,9):
        low, high = np.nanpercentile(data[:,-i],1,interpolation='nearest'), np.nanpercentile(data[:,-i],99,interpolation='nearest')
        data[:,-i] = np.clip(data[:,-i],low,high)
        data[:,-i] = (data[:,-i] - np.min(data[:,-i]))/(np.max(data[:,-i]) - np.min(data[:,-i]))
    train, test = train_test_split(data, test_size=0.3)
    np.save('train',train)
    np.save('test',test)
    train = np.load('train.npy')
    test = np.load('test.npy')
    train = tf.data.Dataset.from_tensor_slices(train).shuffle(10000,reshuffle_each_iteration=True).batch(1000, drop_remainder=True)
    test = tf.data.Dataset.from_tensor_slices(test).shuffle(10000,reshuffle_each_iteration=True).batch(10, drop_remainder=True)
    model = AE()
    optimizer = AdamWeightDecay(learning_rate=1e-3,weight_decay_rate=1e-3)
    checkpoint = tf.train.Checkpoint(model=model)

    @tf.function
    def step(x,training=True):
        with tf.GradientTape() as tape:
            loss = model(x)
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(200):
        total_loss = []
        test_loss = []
        for sample in train:
            total_loss.append(step(sample).numpy())
        for sample in test:
            test_loss.append(step(sample,False).numpy())

        print(epoch+1, np.mean(total_loss), np.mean(test_loss))
        if epoch % 20 == 19:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    main()



