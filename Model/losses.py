"""file to conatin all the loss functions"""
import tensorflow as tf
from tensorflow.keras.losses import  BinaryCrossentropy

cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """discriminator loss function"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    avg_loss = real_loss + fake_loss
    return avg_loss


def generator_loss(fake_output):
    """discriminator loss function"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)
