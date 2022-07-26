import tensorflow as tf
import numpy as np
import time
import os
import argparse

def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_DIMS = [384, 384, 384, 384, 384, 384, 2591]
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in self.G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in self.G_DIMS[:-1]]
        self.output_layer_code = tf.keras.layers.Dense(self.G_DIMS[-1], activation=tf.nn.sigmoid)
        self.output_layer_race = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        x = tf.concat((self.output_layer_race(x),self.output_layer_code(x)),axis=-1)
        return x

    def test(self, x):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=False))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=False))
            x += h
        x = tf.concat((prob2onehot(self.output_layer_race(x)),self.output_layer_code(x)),axis=-1)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_DIMS = [384, 384, 384, 384, 384, 384]
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


def train(modeln):
    checkpoint_directory = "training_checkpoints_emrwgan_"+modeln
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    data = np.load('train.npy').astype('float32')

    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,reshuffle_each_iteration=True).batch(1000, drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)

    generator = Generator()
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
    for epoch in range(15000):
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
        if epoch % 1000 == 999:
            checkpoint.save(file_prefix=checkpoint_prefix)


def gen(modeln):
    checkpoint_directory = "training_checkpoints_emrwgan_" + modeln
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    generator = Generator()

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(checkpoint_prefix + '-15').expect_partial()
    @tf.function
    def g_step():
        z = tf.random.normal(shape=[100, Z_DIM])
        synthetic = generator.test(z)
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
    print(syn.shape)
    data = np.load('covid_vumc.npy').astype('float32')
    for i in range(1,9):
        low, high = np.nanpercentile(data[:,-i],1,interpolation='nearest'), np.nanpercentile(data[:,-i],99,interpolation='nearest')
        data[:,-i] = np.clip(data[:,-i],low,high)
        xmin, xmax = np.min(data[:,-i]),np.max(data[:,-i])
        syn[:, -i] = (1 - syn[:, -i])*xmax + syn[:,-i]*xmin
    np.save('syn/emrwgan_'+modeln, syn)



if __name__ == '__main__':
    batchsize = 1000
    Z_DIM = 128
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str)
    parser.add_argument('a', type=int)
    parser.add_argument('b', type=int)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    for i in range(args.a, args.b):
        train(str(i))
    # gen('5')
    # gen(args.epoch)
    # train('neg')
    # train('x')
    # gen(args.epoch)


