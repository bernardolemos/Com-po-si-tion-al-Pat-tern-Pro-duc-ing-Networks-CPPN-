# -*- coding: utf-8 -*-

import os
import cv2
import time
import random
import argparse
import numpy as np
from skimage.color import hsv2rgb
from scipy.interpolate import interp1d
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import RandomNormal, VarianceScaling


def images_2_gif(image_folder='./seq', video_name='gif', fps=25, loop=1, reverse=True):
    """
    Convert sequence of images to gif
    """
    import moviepy.editor as mpy
    
    #get variables
    video_name = video_name + "_" + str(round(time.time())) + ".gif"

    #Get images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)

    # join paths
    for k in range(len(images)):
        images[k] = os.path.join(image_folder, images[k])

    gif = mpy.ImageSequenceClip(images, fps=fps)
    gif.write_gif(os.path.join(image_folder, video_name), fps=fps)

    return True

def images_2_video(image_folder='./seq', video_name='video', fps=25, loop=1, reverse=True):
    """
    Convert sequence of images to a video
    """    
    #get variables
    video_name = video_name + "_" + str(round(time.time())) + ".avi" 
    
    reverse=True
    if reverse <= 0:
        reverse = False

    #Get images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    #video_format = {}
    #video_format['avi'] = 0
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), 0, fps, (width, height))

    #Generate video
    for l in range(loop):
        if loop > 1 and l > 1 and reverse:
            images = images[::-1]
        for k in range(len(images)):
            video.write(cv2.imread(os.path.join(image_folder, images[k])))

    return True

def create_grid(x_dim=32, y_dim=32, scale=1.0, radius_factor=1.0):
    """
    Create network input (linear space)
    """
    N = np.mean((x_dim, y_dim))
    x = np.linspace(- x_dim / N * scale, x_dim / N * scale, x_dim)
    y = x
    #y = np.linspace(- y_dim / N * scale, y_dim / N * scale, y_dim) # if `x_dim` != `y_dim` 
    # extend vectors to matrices
    X, Y = np.meshgrid(x, y)
    # reshape matrix to array of shape (x_dim*y_dim, 1)
    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    # get radius
    r = np.sqrt(x ** 2 + y ** 2) * radius_factor

    return x, y, r

def interpolate_z(z, seq_len=25, mode=None):
    '''
    Interpolate movement through latent space with spline approximation.
    '''
    x_max = float(z.shape[0])
    if mode is not None:
        x_max += 1
        if 'smooth' in mode:
            x_max += 2
        
    xx = np.arange(0, x_max)
    zt = []
    for k in range(z.shape[1]):
        yy = list(z[:,k])
        if mode is not None:
            yy.append(z[0,k])
            if 'smooth' in mode:
                yy = [z[-1,k]] + yy + [z[1,k]]
        fz = interp1d(xx, yy, kind='cubic')
        if 'smooth' in mode:
            x_new = np.linspace(1, x_max-2, num=seq_len, endpoint=False)
        else:
            x_new = np.linspace(0, x_max-1, num=seq_len, endpoint=False)
        zt.append(fz(x_new))
    
    return np.column_stack(zt)


def create_image(model, x, y, r, z, x_dim, y_dim):
    '''
    create an image for a given latent vector (`z`)
    '''
    # create input vector
    Z = np.repeat(z, x.shape[0]).reshape((-1, x.shape[0]))
    X = np.concatenate([x, y, r, Z.T], axis=1)

    pred = model.predict(X)

    img = []
    n_channels = pred.shape[1]
    for k in range(n_channels):
        yp = pred[:, k]
        # if k == n_channels - 1:
            # yp = np.sin(yp)
        yp = (yp - yp.min()) / (yp.max()-yp.min())
        img.append(yp.reshape(y_dim, x_dim))
        
    img = np.dstack(img)

    if n_channels == 3:
        img = hsv2rgb(img)
        
    return (img * 255).astype(np.uint8)


def create_image_seq(model, x, y, r, z, x_dim, y_dim, seq_len=25, mode=None):
    '''
    create a list of images with `seq_len` between a given latent vectors in `z`
    '''
    # create all z values
    zt = interpolate_z(z, seq_len, mode)
       
    images = []
    for k in range(zt.shape[0]):
        print("Image", k + 1, "of", zt.shape[0])
        images.append(create_image(model, x, y, r, zt[k,:], x_dim, y_dim))
        #sys.stdout.flush()

    return images

def random_normal_init(mean=0.0, variance=1.2):
    '''
    Normal dist. initializer
    '''
    sd = 1.2 ** 0.5 #get standad deviation
    return RandomNormal(mean, sd)

def variance_scaling_intit(variance=1.2):
    '''
    Initializer capable of adapting its scale to the shape of weights
    '''
    return VarianceScaling(scale=variance)

#TODO add other architectures
def build_model(n_units=64, n_hidden_l=2, var=1.2, coloring=True, n_z=16, initializer="vs"):
    """
    Builds Neural Net
    """
    #Init. model
    model = Sequential()

    #input layer
    if initializer == "vs":
        model.add(Dense(n_units, kernel_initializer=variance_scaling_intit(var), input_dim=n_z + 3))
    elif initializer == "normal":
        model.add(Dense(n_units, kernel_initializer=random_normal_init(mean=0.0, variance=var), input_dim=n_z + 3)) #np.sqrt(n_units)
    model.add(Activation('tanh'))

    #hidden layers
    for _ in range(n_hidden_l):
        if initializer == "vs":
            model.add(Dense(n_units, kernel_initializer=variance_scaling_intit(var)))
        elif initializer == "normal":
            model.add(Dense(n_units, kernel_initializer=random_normal_init(mean=0.0, variance=var))) #np.sqrt(n_units)
        model.add(Activation('tanh'))

    #output layer
    model.add(Dense(3 if coloring else 1))
    #Activation('sigmoid'),
    model.add(Activation('linear'))

    model.compile(optimizer='rmsprop', loss='mse')

    return model

########################################################################
########################################################################
def main(args):
    # create file if does not exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    #get variables
    n_z = args.nz
    x_dim = args.dimension
    y_dim = x_dim
    scale = args.scale
    
    coloring = True
    if args.coloring < 1: 
        coloring = False

    x, y, r = create_grid(x_dim=x_dim, y_dim=y_dim, scale=scale, radius_factor=args.radius)
    # in_image_path = "./nyc.jpeg"
    # x, y, r = decompose_image(in_image_path, scale=scale, bw=True)
    #create latent space (random noise)
    z = np.random.normal(0, 1, (3, n_z))

    #create neural network
    model = build_model(n_units=args.nunits, n_hidden_l=args.nhlayers, var=args.variance, coloring=coloring, n_z=n_z, initializer=args.kernel_init)

    st = time.time()

    #Generate images
    #single image
    if args.sequence_len == 1:
        #create images
        print("Creating image...")
        img = create_image(model, x, y, r, z[0,:], x_dim, y_dim)
        # mess with colors
        # img[:,:,0] = img[:,:,0] * 10.1 
        # img[:,:,1] = img[:,:,1] * 0.1
        # img[:,:,2] = img[:,:,2] * 0.3 
        cv2.imwrite(os.path.join(args.path, args.name + "_" + str(round(time.time())) + '.png'), img)
    #sequence of images
    else:
        img_seq = create_image_seq(model, x, y, r, z, x_dim, y_dim, seq_len=args.sequence_len, mode='smooth')
        sl_aux = len(str(args.sequence_len))
        for k, img in enumerate(img_seq):
            k_str = str(k+1)
            pre = "0" * (sl_aux - len(k_str))
            suffix = pre + str(k)
            cv2.imwrite(os.path.join(args.path, args.name + "_" + suffix + ".png"), img)

        # generate video
        print("Generating video...")
        images_2_video(image_folder=args.path, video_name=args.vname, fps=args.framesps, loop=args.loop, reverse=args.reverse)
        if args.gif == 1:
            images_2_gif(image_folder=args.path, fps=args.framesps, loop=args.loop, reverse=args.reverse)

    print("Total time:", time.time() - st, "sec.")

    return 0

if __name__ == "__main__":
    # TODO conditions
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--sequence_len', type=int, nargs='?', default=1)
    parser.add_argument('-c', '--coloring', type=int, nargs='?', default=1)
    parser.add_argument('-p', '--path', type=str, nargs='?', default="./images")
    parser.add_argument('-d', '--dimension', type=int, nargs='?', default=720)
    parser.add_argument('-v', '--variance', type=float, nargs='?', default=1.5)
    parser.add_argument('-fps', '--framesps', type=int, nargs='?', default=25)
    parser.add_argument('-nz', type=int, nargs='?', default=16)
    parser.add_argument('-sc', '--scale', type=float, nargs='?', default=5.0)
    parser.add_argument('-nhl', '--nhlayers', type=int, nargs='?', default=2)
    parser.add_argument('-nu', '--nunits', type=int, nargs='?', default=32)
    parser.add_argument('-nm', '--name', type=str, nargs='?', default="image")
    parser.add_argument('-lp', '--loop', type=int, nargs='?', default=1)
    parser.add_argument('-rv', '--reverse', type=int, nargs='?', default=1)
    parser.add_argument('-vn', '--vname', type=str, nargs='?', default="video")
    parser.add_argument('-rd', '--radius', type=float, nargs='?', default=1.0)
    parser.add_argument('-ki', '--kernel_init', type=str, nargs='?', default="vs")
    parser.add_argument('-g', '--gif', type=int, nargs='?', default=0)
    args = parser.parse_args()
    
    import sys

    # update name
    if len(sys.argv) > 1:
        skip = False
        for i in range(1, len(sys.argv)):
            # skip path argument
            if skip == True:
                skip = False
                continue
            if sys.argv[i] == "-p" or sys.argv[i] == "--path":
                skip = True
                continue
            args.name += "_" + sys.argv[i] 
            args.vname += "_" + sys.argv[i] 

    main(args)




