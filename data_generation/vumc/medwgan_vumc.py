import tensorflow as tf
import numpy as np
import time
import os, argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)


class AE(tf.keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.decoder0 = tf.keras.layers.Dense(5, activation=tf.nn.softmax,
                                              kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                              bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))
        self.decoder1 = tf.keras.layers.Dense(2591, activation=tf.nn.sigmoid,
                                              kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                              bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))

    def call(self, latent):
        x_hat = tf.concat((self.decoder0(latent), self.decoder1(latent)), axis=-1)
        return x_hat

    def test(self, latent):
        x_hat = tf.concat((prob2onehot(self.decoder0(latent)), self.decoder1(latent)), axis=-1)
        return x_hat


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_DIMS = [384, 384, 384, 384, 384]
        self.dense_layers = [tf.keras.layers.Dense(dim, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5)) for dim in
                             self.G_DIMS]
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization(epsilon=1e-5, beta_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                               gamma_regularizer=tf.keras.regularizers.L2(l2=2.5e-5)) for _ in
            self.G_DIMS]

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1, len(self.G_DIMS[:-1])):
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
        self.D_DIMS = [384, 384, 384, 384, 384, 384]
        self.dense_layers = [
            tf.keras.layers.Dense(dim, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                  bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))
            for dim in self.D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                  bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))

    def call(self, x):
        x = tf.concat((x, tf.tile(tf.reduce_mean(x, axis=0, keepdims=True), [batchsize, 1])), axis=-1)
        for i in range(len(self.D_DIMS)):
            x = self.dense_layers[i](x)
        x = self.output_layer(x)
        return x


def train(model):
    checkpoint_directory_ae = "training_checkpoints_ae"
    checkpoint_prefix_ae = os.path.join(checkpoint_directory_ae, "ckpt")
    checkpoint_directory = "training_checkpoints_medwgan_new_" + model
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('train.npy')
    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000, reshuffle_each_iteration=True).batch(1000,
                                                                                                                 drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-5)

    generator = Generator()
    discriminator = Discriminator()
    ae = AE()
    checkpoint = tf.train.Checkpoint(generator=generator, ae=ae)
    checkpoint_ae = tf.train.Checkpoint(model=ae)
    checkpoint_ae.restore(checkpoint_prefix_ae + '-4').expect_partial()

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[batchsize, Z_DIM])

        with tf.GradientTape() as disc_tape:
            synthetic = ae(generator(z, False))

            real_output = discriminator(real)
            fake_output = discriminator(synthetic)

            disc_loss = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)) + tf.reduce_sum(
                discriminator.losses)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        for w in discriminator.trainable_variables:
            w.assign(tf.clip_by_value(w, -0.01, 0.01))
        return disc_loss

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = ae(generator(z, True))

            fake_output = discriminator(synthetic)

            gen_loss = -tf.reduce_mean(fake_output) + tf.reduce_sum(generator.losses) + tf.reduce_sum(ae.losses)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables + ae.trainable_variables)
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables + ae.trainable_variables))

    for epoch in range(15000):
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
            print(format_str % (epoch, -total_loss / step, duration_epoch))
        if epoch % 1000 == 999:
            checkpoint.save(file_prefix=checkpoint_prefix)


def gen(model):
    checkpoint_directory = "training_checkpoints_medwgan_new_" + model
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    generator = Generator()
    ae = AE()
    checkpoint = tf.train.Checkpoint(generator=generator, ae=ae)
    checkpoint.restore(checkpoint_prefix + '-15').expect_partial()

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
    np.save('syn_medwgan/medwgan_'+model, syn)


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
    # for s in range(20):
    #     train(str(s))
    #     print(s)

