from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D, SeparableConv2D, Conv2D
from tensorflow.keras.activations import relu

from util import use_gpu

from imutils import paths

import numpy as np

from sklearn.model_selection import train_test_split

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Hair Mask Segmentation CNN Train')
    parser.add_argument('--image_data_format', default='channels_last')
    parser.add_argument('--mini_batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=5)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--set_size', type=int, default=200)
    parser.add_argument('--model_name', type=str, default='./new_model.h5')
    parser.add_argument('--training_data_path',  default='./CelebA-HQ-img/')
    parser.add_argument('--testing_data_path',  default='./CelebAMask-HQ-mask-anno/')
    parser.add_argument('--gpu_mem', type=int, default=2)
    args = parser.parse_args()

    return args

def setup_model(args):
    
    baseModel = MobileNet(weights="imagenet", include_top=False,
	    input_tensor=Input(shape=(224, 224, 3)))

    for layer in baseModel.layers:
	    layer.trainable = False

    respoints = (baseModel.layers[9], baseModel.layers[19], baseModel.layers[35], baseModel.layers[72])

    headModel = baseModel.output

    headModel = UpSampling2D(size=(2, 2), data_format=args.image_data_format, name='upsampling_14')(headModel)
    headModel = Concatenate(axis=-1)([headModel, respoints[3].output])
    headModel = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=relu, data_format=args.image_data_format, name='upsampling_dw_conv3x3_15')(headModel)

    headModel = UpSampling2D(size=(2, 2), data_format=args.image_data_format, name='upsampling_16')(headModel)
    headModel = Concatenate(axis=-1)([headModel, respoints[2].output])
    headModel = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=relu, data_format=args.image_data_format, name='upsampling_dw_conv3x3_17')(headModel)

    headModel = UpSampling2D(size=(2, 2), data_format=args.image_data_format, name='upsampling_18')(headModel)
    headModel = Concatenate(axis=-1)([headModel, respoints[1].output])
    headModel = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=relu,  data_format=args.image_data_format, name='upsampling_dw_conv3x3_19')(headModel)

    headModel = UpSampling2D(size=(2, 2), data_format=args.image_data_format, name='upsampling_20')(headModel)
    headModel = Concatenate(axis=-1)([headModel, respoints[0].output])
    headModel = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=relu,  data_format=args.image_data_format, name='upsamplingd_dw_conv3x3_21')(headModel)

    headModel = UpSampling2D(size=(2, 2), data_format=args.image_data_format, name='upsampling_22')(headModel)
    headModel = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=relu,  data_format=args.image_data_format, name='upsampling_dw_conv3x3_23')(headModel)

    headModel = Conv2D(filters=1, kernel_size=(1,1), padding='same',   data_format=args.image_data_format, name='conv_1x1_softmax')(headModel)

    return Model(inputs=baseModel.input, outputs=headModel)

def train(model, args):

    training_cycles = int(args.samples/args.set_size)
    for i in range(0, training_cycles):
        imageData = []
        maskData = []

        beginning_of_mini_set = int(i*args.set_size-(args.set_size/2))
        end_of_mini_set = int(args.set_size+(i*args.set_size)-(args.set_size/2))
        
        for j in range(beginning_of_mini_set, end_of_mini_set):
            try :

                maskData.append(img_to_array(load_img(f"{args.testing_data_path}{j}.png", color_mode="grayscale", target_size=(224, 224))))
                imageData.append(img_to_array(load_img(f"{args.training_data_path}{j}.jpg", target_size=(224, 224))))

            except FileNotFoundError as error:

                continue

        maskData = np.array(maskData, dtype="float32")
        imageData = np.array(imageData, dtype="float32")


        imageData = imageData / 255.0
        maskData = maskData / 255
        maskData[maskData > 0.5] = 1
        maskData[maskData <= 0.5] = 0

        (trainX, testX, trainY, testY) = train_test_split(imageData, maskData)
        

        model.fit(
            x=trainX, y=trainY, batch_size=args.mini_batch_size,
            validation_data=(testX, testY),
            steps_per_epoch = args.steps_per_epoch,
            epochs=args.epochs)


if __name__ == '__main__':

    args = get_args()
    use_gpu(args.gpu_mem)
    model = setup_model(args)
    opt = Adam(lr=args.lr, decay=args.lr/args.epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt,
	    metrics=["accuracy"])
    train(model, args)
    model.save(args.model_name)
