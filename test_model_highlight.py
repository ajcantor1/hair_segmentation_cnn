from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from util import use_gpu

import cv2

import numpy as np

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Hair Mask Segmentation Highlight Test')
    parser.add_argument('--model', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_image_path', type=str)
    parser.add_argument('--gpu_mem', type=int, default=2)
    args = parser.parse_args()

    return args

def validate_args(args):

    error_str = ""

    if not args.model:
        error_str = "No model file was supplied. "

    if not args.image_path: 
        error_str = error_str + "No image was provided. "

    print(error_str)
    return (len(error_str)==0)

def test(model, args): 

    testX = img_to_array(load_img(args.image_path, target_size=(224, 224)))
    testX = testX / 255.0
    predIdxs = model.predict(np.array( [testX,] ))
    result = np.squeeze(predIdxs)
    result[result<=args.threshold] = 0
    result = result * 255
    result = result.astype('uint8')

    image = cv2.imread(args.image_path, -1)
    image = cv2.resize(image, result.shape, interpolation = cv2.INTER_AREA)

    contours,hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    blue,green,red = cv2.split(image)

    red = cv2.add(red, 200, dst = red, mask = result)
    cv2.merge((blue, green, red), image)
    
    cv2.imshow('image', image)
    
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    if args.save_image_path:
        cv2.imwrite(args.save_image_path, image)

if __name__ == '__main__':

    args = get_args()
    use_gpu(args.gpu_mem)
    
    if validate_args(args):
        
        model = load_model(args.model)
        test(model, args)


