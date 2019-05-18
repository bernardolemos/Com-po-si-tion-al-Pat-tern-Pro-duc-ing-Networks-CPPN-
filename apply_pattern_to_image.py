import os
import cv2
import time
import argparse
from PIL import Image


def main(args):
    # create output direcotry if does not exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    name = "imagep_"

    # get paterns
    patterns = [pattern for pattern in os.listdir(args.patterns_path) if pattern.endswith(".png")]
    patterns = sorted(patterns)

    # read input image
    in_image = cv2.imread(args.image_path)
    # resize
    frame = cv2.imread(os.path.join(args.patterns_path, patterns[0]))
    dim, _, _ = frame.shape
    in_image = cv2.resize(in_image, (dim, dim))

    for p_path in patterns:
        # read pattern
        pattern = cv2.imread(os.path.join(args.patterns_path, p_path))
        # apply pattern
        img_p = Image.fromarray(in_image + pattern)
        # save image pattern
        img_p.save(os.path.join(args.out_path, name + str(round(time.time()))  + ".png"))

if __name__ == "__main__":
    # TODO conditions
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image_path', type=str, nargs='?', default="./")
    parser.add_argument('-pp', '--patterns_path', type=str, nargs='?', default="./")
    parser.add_argument('-op', '--out_path', type=str, nargs='?', default="./")
    args = parser.parse_args()

    main(args)  