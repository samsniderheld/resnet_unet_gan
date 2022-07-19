"""The file where we build our resnet based generator
as well as our discriminator"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout,Add
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications import ResNet50

from Model.layers import SelfAttention

def decoder_block(input, skip_features, num_filters,attn=False):
    """base decoder block for the unet decoder"""
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", use_bias=False)(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    if attn:
        x = SelfAttention(num_filters)(x)

    return x

def conv_block(input, num_filters):
    """base convolutional block for the decoder"""
    x = Conv2D(num_filters, 3, padding="same", use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_resnet50_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    resnet50.trainable = False
    # resnet50.summary()

    """ Encoder """
    s_1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s_2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s_3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s_4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)


    """ Bridge """
    b_1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d_1 = decoder_block(b_1, s_4, 256,True)                     ## (64 x 64)
    d_2 = decoder_block(d_1, s_3, 128)                     ## (128 x 128)
    d_3 = decoder_block(d_2, s_2, 64)                     ## (256 x 256)
    d_4 = decoder_block(d_3, s_1, 32)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(3, 1, padding="same")(d_4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")

    return model
                 