import argparse
import cv2
import os

from networks.style_transfer_network import StyleTransfer
from utils.config import Flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/cubist.yml', help='the path to the conf file')
    parser.add_argument('-i', '--image', default='img/sin_character.jpg', help='the test image')
    parser.add_argument('-o', '--origin_color', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    style_name = os.path.splitext(os.path.split(args.conf)[-1])[0]

    FLAGS = Flag.read_config(args.conf)
    image = cv2.imread(args.image)

    model = StyleTransfer(FLAGS)
    styled_img = model.evaluate_network(args.image, image.shape[:2], origin_color=args.origin_color)
    styled_img = cv2.cvtColor(styled_img, cv2.COLOR_RGB2BGR)

    save_image_path = list(os.path.split(args.image))
    save_image_path[-1] = style_name + "_styled_" + save_image_path[-1]
    cv2.imwrite("/".join(save_image_path), styled_img)
