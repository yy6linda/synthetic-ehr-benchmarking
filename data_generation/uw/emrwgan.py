import tensorflow as tf
import numpy as np
import time
import os,re
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)


class Generator(tf.keras.Model):
    def __init__(self, size):
        super(Generator, self).__init__()
        self.G_DIMS = [384,384,384,384,384,384, size]
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in self.G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in self.G_DIMS[:-1]]
        self.output_layer_code = tf.keras.layers.Dense(self.G_DIMS[-1])

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        x = self.output_layer_code(x)
        race = tf.nn.softmax(tf.concat((x[:,2596:2599], x[:, 2600:2603]),axis=-1))
        x = tf.concat((tf.nn.sigmoid(x[:,:2596]), race[:,:3], tf.expand_dims(tf.nn.sigmoid(x[:, 2599]),-1), race[:,3:], tf.nn.sigmoid(x[:, 2603:])),axis=-1)
        return x

    def test(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        x = self.output_layer_code(x)
        race = prob2onehot(tf.concat((x[:,2596:2599], x[:, 2600:2603]),axis=-1))
        x = tf.concat((tf.nn.sigmoid(x[:,:2596]), race[:,:3], tf.expand_dims(tf.nn.sigmoid(x[:, 2599]),-1), race[:,3:], tf.nn.sigmoid(x[:, 2603:])),axis=-1)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_DIMS = [384,384,384,384,384,384]
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu) for dim in self.D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in self.D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense_layers[0](x)
        x = self.layer_norm_layers[0](x)
        for i in range(1,len(self.D_DIMS)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        x = self.output_layer(x)
        return x


def train(model, k):
    checkpoint_directory = "training_checkpoints_emrwgan_new_" + model + '_' + k
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('data/train_' + str(model) + '.npy').astype('float32')
    np.random.shuffle(data)
    size = data.shape[-1]
    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,reshuffle_each_iteration=True).batch(1000, drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)

    generator = Generator(size)
    discriminator = Discriminator()

    checkpoint = tf.train.Checkpoint(generator=generator)

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[batchsize, Z_DIM])

        epsilon = tf.random.uniform(
            shape=[batchsize, 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, False)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real)
            fake_output = discriminator(synthetic)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch(interpolate)
                interpolate_output = discriminator(interpolate)
            w_grad = t.gradient(interpolate_output, interpolate)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            disc_loss = 10 * gradient_penalty + w_distance

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z,True)

            fake_output = discriminator(synthetic)

            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        disc_loss, w_distance = d_step(batch)
        g_step()
        return disc_loss, w_distance

    print('training start')
    if model != 'pos':
        for epoch in range(2000):
            start_time = time.time()
            total_loss = 0.0
            total_w = 0.0
            step = 0.0
            for args in dataset_train:
                loss, w = train_step(args)
                total_loss += loss
                total_w += w
                step += 1
            duration_epoch = time.time() - start_time
            format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
            if epoch % 10 == 9:
                print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch))
            if epoch % 100 == 99:
                checkpoint.save(file_prefix=checkpoint_prefix)

    else:
        for epoch in range(30000):
            start_time = time.time()
            total_loss = 0.0
            total_w = 0.0
            step = 0.0
            for args in dataset_train:
                loss, w = train_step(args)
                total_loss += loss
                total_w += w
                step += 1
            duration_epoch = time.time() - start_time
            format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
            if epoch % 250 == 249:
                print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch))
            if epoch % 2000 == 1999:
                checkpoint.save(file_prefix=checkpoint_prefix)


def gen(model, e, n):
    checkpoint_directory = "training_checkpoints_emrwgan_new_" + model + '_' + n
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('data/train_' + str(model) + '.npy').astype('float32')
    size = data.shape[-1]
    generator = Generator(size)
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_prefix + '-' + e).expect_partial()

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[100, Z_DIM])
        synthetic = generator(z, False)
        return synthetic

    if model != 'combined':
        syn = []
        for _ in range(int(data.shape[0]*1.5/100)):
            syn.extend(np.round(g_step().numpy()))
        syn = np.array(syn, dtype='int8')
    else:
        pos = int(np.sum(data[:, -1] == 1) * 1.5)
        neg = int(np.sum(data[:, -1] == 0) * 1.5)
        syn_pos = []
        syn_neg = []
        while len(syn_pos) <= pos:
            tmp = g_step().numpy()
            syn_pos.extend(np.round(tmp[tmp[:, -1] >= 0.5]))
        while len(syn_neg) <= neg:
            tmp = g_step().numpy()
            syn_neg.extend(np.round(tmp[tmp[:, -1] < 0.5]))
        syn = np.array(syn_pos + syn_neg, dtype='int8')
        np.random.shuffle(syn)
    np.save('syn_emrwgan/emrwgan_' + model + '_' + n, syn)
    plt.figure(figsize=(10,10))
    plt.scatter(np.mean(np.round(data),axis=0),np.mean(syn,axis=0))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('test'+n+model, dpi=100)



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

