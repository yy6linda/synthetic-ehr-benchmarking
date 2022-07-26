import tensorflow as tf
import numpy as np
import time
import os, re,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)


class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.decoder0 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.decoder1 = tf.keras.layers.Dense(2591, activation=tf.nn.sigmoid)

    def call(self, latent):
        x_hat = tf.concat((self.decoder0(latent), self.decoder1(latent)), axis=-1)
        return x_hat

    def test(self, latent):
        x_hat = tf.concat((prob2onehot(self.decoder0(latent)), self.decoder1(latent)), axis=-1)
        return x_hat


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_DIMS = [384,384,384,384,384]
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in self.G_DIMS]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in self.G_DIMS]

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        h = self.dense_layers[-1](x)
        h = tf.nn.tanh(self.batch_norm_layers[-1](h, training=training))
        x += h
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_DIMS = [384,384,384,384,384,384]
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu)
                             for dim in self.D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

    def call(self, x):
        x = tf.concat((x,tf.tile(tf.reduce_mean(x,axis=0,keepdims=True),[batchsize,1])),axis=-1)
        for i in range(len(self.D_DIMS)):
            x = self.dense_layers[i](x)
        x = self.output_layer(x)
        return x


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


def train(model):
    checkpoint_directory_ae = "training_checkpoints_ae"
    checkpoint_prefix_ae = os.path.join(checkpoint_directory_ae, "ckpt")
    checkpoint_directory = "training_checkpoints_medbgan_" + model
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('train.npy')
    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,reshuffle_each_iteration=True).batch(1000, drop_remainder=True)

    generator_optimizer = AdamWeightDecay(learning_rate=1e-5, weight_decay_rate=0.001)
    discriminator_optimizer = AdamWeightDecay(learning_rate=1e-5, weight_decay_rate=0.001)

    generator = Generator()
    discriminator = Discriminator()
    ae = AE()

    checkpoint = tf.train.Checkpoint(generator=generator,ae=ae)
    checkpoint_ae = tf.train.Checkpoint(model=ae)
    checkpoint_ae.restore(checkpoint_prefix_ae + '-4').expect_partial()

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[batchsize, Z_DIM])

        with tf.GradientTape() as disc_tape:
            synthetic = ae(generator(z, False))

            real_output = discriminator(real)
            fake_output = discriminator(synthetic)

            disc_loss = -tf.reduce_mean(tf.math.log(real_output+1e-12)) - tf.reduce_mean(tf.math.log(1-fake_output+1e-12))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = ae(generator(z, True))

            fake_output = discriminator(synthetic)
            # gen_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-12))
            gen_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.math.log(fake_output + 1e-12) - tf.math.log(1. - fake_output + 1e-12)))

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables + ae.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables + ae.trainable_variables))

    for epoch in range(3000):
        start_time = time.time()
        total_loss = 0.0
        step = 0.0
        for batch_sample in dataset_train:
            for _ in range(2):
                loss = d_step(batch_sample)
                total_loss += loss
            step += 1
            g_step()
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f (%.2f)'
        if epoch % 250 == 249:
            print(format_str % (epoch, total_loss / step * 1000, duration_epoch))
        if epoch % 1000 == 999:
            checkpoint.save(file_prefix=checkpoint_prefix)


def gen(model):
    checkpoint_directory = "training_checkpoints_medbgan_" + model
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    generator = Generator()
    ae = AE()
    checkpoint = tf.train.Checkpoint(generator=generator, ae=ae)
    checkpoint.restore(checkpoint_prefix + '-1').expect_partial()

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[100, Z_DIM])
        synthetic = ae.test(generator(z, False))
        return synthetic

    data = np.load('train.npy').astype('float32')
    pos = int(np.sum(data[:,5] == 1)*1.5)
    neg = int(np.sum(data[:,5] == 0)*1.5)
    syn_pos = []
    syn_neg = []
    while len(syn_pos) <= pos:
        tmp = g_step().numpy()
        syn_pos.extend(tmp[tmp[:,5] >= 0.5])
    while len(syn_neg) <= neg:
        tmp = g_step().numpy()
        syn_neg.extend(tmp[tmp[:,5] < 0.5])
    syn = np.array(syn_pos+syn_neg)
    data = np.load('covid_vumc.npy').astype('float32')
    for i in range(1,9):
        low, high = np.nanpercentile(data[:,-i],1,interpolation='nearest'), np.nanpercentile(data[:,-i],99,interpolation='nearest')
        data[:,-i] = np.clip(data[:,-i],low,high)
        xmin, xmax =np.min(data[:,-i]),np.max(data[:,-i])
        syn[:, -i] = (1-syn[:, -i])*xmax + syn[:,-i]*xmin
    np.save('syn/medbgan_'+model, syn)
    # plt.figure(figsize=(10,10))
    # plt.scatter(np.mean(syn,axis=0),np.mean(data,axis=0))
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.savefig('medbgan', dpi=100)



if __name__ == '__main__':
    batchsize = 1000
    Z_DIM = 128
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    for i in range(5):
        gen(str(i))