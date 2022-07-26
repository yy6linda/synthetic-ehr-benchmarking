import tensorflow as tf
import numpy as np
import time
import os, argparse
import tensorflow.keras.backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prob2onehot(prob):
    return tf.one_hot(tf.math.argmax(prob, axis=-1), depth=6, dtype=tf.float32)


class AE(tf.keras.Model):
    def __init__(self,size):
        super(AE, self).__init__()
        self.decoder = tf.keras.layers.Dense(size, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))

    def decode(self,latent):
        x = self.decoder(latent)
        race = tf.nn.softmax(tf.concat((x[:, 2596:2599], x[:, 2600:2603]), axis=-1))
        x_hat = tf.concat((tf.nn.sigmoid(x[:, :2596]), race[:, :3], tf.expand_dims(tf.nn.sigmoid(x[:, 2599]), -1),
                           race[:, 3:], tf.nn.sigmoid(x[:, 2603:])), axis=-1)
        return x_hat

    def decode_test(self, latent):
        x = self.decoder(latent)
        race = prob2onehot(tf.concat((x[:,2596:2599], x[:, 2600:2603]),axis=-1))
        x_hat = tf.concat((tf.nn.sigmoid(x[:, :2596]), race[:,:3], tf.expand_dims(tf.nn.sigmoid(x[:, 2599]),-1), race[:,3:], tf.nn.sigmoid(x[:, 2603:])),axis=-1)
        return x_hat


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5)) for dim in G_DIMS]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5, beta_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   gamma_regularizer=tf.keras.regularizers.L2(l2=2.5e-5)) for _ in G_DIMS]

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(G_DIMS[:-1])):
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
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))
                             for dim in D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2=2.5e-5),
                                                   bias_regularizer=tf.keras.regularizers.L2(l2=2.5e-5))

    def call(self, x):
        # x = tf.concat((x,tf.tile(tf.reduce_mean(x,axis=0,keepdims=True),[batchsize,1])),axis=-1)
        for i in range(len(D_DIMS)):
            x = self.dense_layers[i](x)
        x = self.output_layer(x)
        return x


class NoisyOptimizer(tf.keras.optimizers.RMSprop):
    def __init__(self, learning_rate):
        super(NoisyOptimizer, self).__init__(learning_rate=learning_rate)

    def get_gradients(self, loss, params):
        grads = super(NoisyOptimizer, self).get_gradients(loss, params)

        grads = [
            grad + K.random_normal(
                grad.shape.as_list(),
                mean=0.0,
                stddev=float(args.sigma) * 8000.0,
                dtype=K.dtype(grads[0])
            )
            for grad in grads
        ]
        return grads


def train(model, s):
    checkpoint_directory_ae = "training_checkpoints_ae_" + model
    checkpoint_prefix_ae = os.path.join(checkpoint_directory_ae, "ckpt")
    checkpoint_directory = "training_checkpoints_dpgan_new"+str(int(float(args.sigma)*100000))+"_" + model + '_'+s
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('data/train_' + str(model) + '.npy').astype('float32')
    np.random.shuffle(data)
    size = data.shape[-1]
    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,reshuffle_each_iteration=True).batch(1000, drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-5)

    generator = Generator()
    discriminator = Discriminator()
    ae = AE(size)

    checkpoint = tf.train.Checkpoint(generator=generator,ae=ae)
    checkpoint_ae = tf.train.Checkpoint(model=ae)
    checkpoint_ae.restore(checkpoint_prefix_ae + '-1').expect_partial()

    # @tf.function
    # def get_mgd(real, synthetic):
    #     states = tf.TensorArray(tf.float32, size=batchsize)
    #     for n_sample in tf.range(batchsize):
    #         real_output = tf.expand_dims(real[n_sample], 0)
    #         fake_output = tf.expand_dims(synthetic[n_sample], 0)
    #         with tf.GradientTape() as disc_tape:
    #             real_output = tf.squeeze(discriminator(real_output))
    #             fake_output = tf.squeeze(discriminator(fake_output))
    #             disc_loss = (-real_output + fake_output)
    #         gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    #         states = states.write(n_sample, tf.linalg.global_norm(gradients_of_discriminator))
    #     return states.stack()

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[batchsize, Z_DIM])

        with tf.GradientTape() as disc_tape:
            synthetic = ae.decode(generator(z, False))

            real_output = discriminator(real)
            fake_output = discriminator(synthetic)

            disc_loss = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)) + tf.reduce_sum(discriminator.losses)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gradients_of_discriminator = [g + K.random_normal(
                    g.shape.as_list(),
                    mean=0.0,
                    stddev=float(args.sigma) * 8000.,
                    dtype=K.dtype(g)) / 1000.0 for g in gradients_of_discriminator]
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        for w in discriminator.trainable_variables:
            w.assign(tf.clip_by_value(w, -0.01, 0.01))
        # if test_gd_norm:
        #     states = tf.reduce_max(get_mgd(real, synthetic))
        # else:
        #     states = tf.constant(0.0)
        return disc_loss

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = ae.decode(generator(z, True))

            fake_output = discriminator(synthetic)

            gen_loss = -tf.reduce_mean(fake_output) + tf.reduce_sum(generator.losses) + tf.reduce_sum(ae.losses)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables + ae.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables + ae.trainable_variables))

    if model != 'pos':
        for epoch in range(1500):
            start_time = time.time()
            total_loss = 0.0
            step = 0.0
            for n_step, batch_sample in enumerate(dataset_train):
                for _ in range(2):
                    loss = d_step(batch_sample)
                    total_loss += loss.numpy()
                step += 1
                g_step()
            duration_epoch = time.time() - start_time
            format_str = 'epoch: %d, loss = %f (%.2f)'
            if epoch % 10 == 9:
                print(format_str % (epoch, -total_loss / step, duration_epoch))
            if epoch % 100 == 99:
                checkpoint.save(file_prefix=checkpoint_prefix)
    else:
        for epoch in range(20000):
            start_time = time.time()
            total_loss = 0.0
            step = 0.0
            for n_step, batch_sample in enumerate(dataset_train):
                for _ in range(2):
                    loss = d_step(batch_sample)
                    total_loss += loss.numpy()
                step += 1
                g_step()
            duration_epoch = time.time() - start_time
            format_str = 'epoch: %d, loss = %f (%.2f)'
            if epoch % 200 == 199:
                print(format_str % (epoch, -total_loss / step, duration_epoch))
            if epoch % 2000 == 1999:
                checkpoint.save(file_prefix=checkpoint_prefix)


def gen(model, e, s):
    checkpoint_directory = "training_checkpoints_dpgan_new"+str(int(float(args.sigma)*100000))+"_" + model + '_'+s
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('data/train_' + str(model) + '.npy').astype('float32')
    size = data.shape[-1]
    generator = Generator()
    ae = AE(size)
    checkpoint = tf.train.Checkpoint(generator=generator, ae=ae)
    checkpoint.restore(checkpoint_prefix + '-' + e).expect_partial()

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[1000, Z_DIM])
        synthetic = ae.decode_test(generator(z, False))
        return synthetic

    if model != 'combined':
        syn = []
        for _ in range(int(data.shape[0]*1.5/1000)):
            syn.extend(np.round(g_step().numpy()))
        syn = np.array(syn)
            # syn.extend(g_step().numpy())
    else:
        pos = int(np.sum(data[:, -1] == 1) * 1.5)
        neg = int(np.sum(data[:, -1] == 0) * 1.5)
        syn_pos = []
        syn_neg = []
        while len(syn_pos) <= pos:
            tmp = g_step().numpy()
            tmp = np.round(tmp[tmp[:, -1] >= 0.5])
            if len(tmp) < 1:return
            print(len(tmp))
            syn_pos.extend(tmp)
        while len(syn_neg) <= neg:
            tmp = g_step().numpy()
            syn_neg.extend(np.round(tmp[tmp[:, -1] < 0.5]))
        syn = np.array(syn_pos + syn_neg)
        np.random.shuffle(syn)
    np.save('syn_dpgan_00001/dpgan_checkpoint' + e + '_' + model + '_' + s, syn.astype('int8'))
    plt.figure(figsize=(10, 10))
    plt.scatter(np.mean(syn, axis=0), np.mean(data, axis=0))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('syn_dpgan_00001/' + model + s + '_' + e, dpi=100)


if __name__ == '__main__':
    batchsize = 1000
    Z_DIM = 128
    G_DIMS = [384, 384, 384, 384, 384]
    D_DIMS = [384, 384, 384, 384, 384, 384]
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    parser.add_argument('sigma', type=str)
    parser.add_argument('m', type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

