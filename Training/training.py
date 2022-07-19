"""
File that defines the main training loops.
Called from the main file image_2_image.py
"""
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from tqdm import tqdm

from Data_Utils.generate_samples import get_random_sample
from Data_Utils.data_generator import DataGeneratorGen, DataGeneratorDisc
from Model.unet import build_resnet50_unet
from Model.discriminator import build_discriminator
from Model.losses import discriminator_loss, generator_loss
from Utils.reporting import generate_images, save_experiment_history

def pre_train_unet(args):
    """wrapper function to pretain our uent"""
    res = args.img_dim
    input_shape = (res,res, 3)
    random_sample = get_random_sample(img_size=input_shape[0])
    batch_size = args.batch_size
    data_generator = DataGeneratorGen(img_dims=input_shape[0],
        batch_size=batch_size,data_size=args.data_size)

    generator = build_resnet50_unet(input_shape)
    generator.summary()

    generator.compile(optimizer="adam", loss=MeanSquaredError())

    img_path = args.save_img_path
    model_path = args.gen_pre_train_model_path_frozen

    print("train decoder")

    for i in range(args.gen_pre_train_epochs):

        print(f'training epoch {str(i)}')

        generator.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        generate_images(generator,random_sample, img_path, i)


        generator.save_weights(model_path)

    img_path = args.save_img_path
    model_path = args.gen_pre_train_model_path_unfrozen

    generator.trainable = True
    generator.summary()

    print("train encoder and decoder")

    for i in range(args.gen_pre_train_epochs,args.gen_pre_train_epochs+3):

        print(f'training epoch {str(i)}')

        generator.fit(
            data_generator,
            epochs=1,
            shuffle=True,
        )

        generate_images(generator,random_sample, img_path, i)


    generator.save_weights(model_path)

def pre_train_discriminator(args):
    """run the discriminator pretrainng phase"""
    discriminator = build_discriminator()
    discriminator.summary()

    res = args.img_dim
    batch_size = args.batch_size

    data_generator_disc = DataGeneratorDisc(img_dims=res,
        batch_size=batch_size)

    all_disc_loss = []

    learning_rate = args.lr

    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in tqdm(range(args.disc_pre_train_epochs)):
        num_batches = data_generator_disc.count

        for i in range(0,num_batches):

            batch = data_generator_disc.__getitem__(i)
            disc_loss = disc_train_step(batch, discriminator, 
                discriminator_optimizer)

            all_disc_loss.append(disc_loss)

    model_path = args.disc_pre_train_model_path
    discriminator.save_weights(model_path)


@tf.function
def disc_train_step(images,discriminator, optimizer):
    """training step for disc g"""
    with tf.GradientTape() as disc_tape:

        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(images[1], training=True)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

    optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

    return disc_loss

@tf.function
def gen_train_step(images, generator, discriminator, gen_opt):
    """training step fo generator"""
    with tf.GradientTape() as gen_tape:
        generated_images = generator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))

    return gen_loss



def train_unet_gan(args):
    """the training loop for the main gan"""
    img_path = args.save_img_path

    batch_size = args.batch_size

    input_shape = (args.img_dim,args.img_dim, 3)

    random_sample = get_random_sample(img_size=input_shape[0])
  
    data_generator = DataGeneratorGen(img_dims=input_shape[0],
        batch_size=batch_size,data_size=args.data_size)

    generator = build_resnet50_unet(input_shape)
    generator.summary()

    discriminator = build_discriminator()
    discriminator.summary()

    generator.load_weights(args.gen_pre_train_model_path_unfrozen)
    generator.trainable = True
    discriminator.load_weights(args.disc_pre_train_model_path)
    discriminator.trainable = True

    generator_optimizer = tf.keras.optimizers.Adam(args.lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.lr)

    #setup reporting lists
    all_disc_loss = []
    all_gen_loss = []

    train_gen = False
    critic_thresh= args.critic_thresh


    for epoch in tqdm(range(args.gan_epochs)):
        print(f"epoch {str(epoch)}")

        num_batches = data_generator.count

        for i in tqdm(range(0,num_batches)):
            batch = data_generator.__getitem__(i)

            if train_gen:
                gen_loss = gen_train_step(batch, generator, 
                    discriminator, generator_optimizer)
                all_gen_loss.append(gen_loss)
                train_gen = False

            else:
                disc_loss = disc_train_step(batch, discriminator, discriminator_optimizer)
                all_disc_loss.append(disc_loss)

            if disc_loss<critic_thresh:
                train_gen = True


    model_path = args.final_model_path
    generator.save_weights(model_path)

