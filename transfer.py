import argparse

import tensorflow as tf

from networks.style_transfer_network import StyleTransfer
from trainers.style_transfer_trainer import StyleTransferTrainer
from utils.config import Flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    FLAGS = Flag.read_config(args.conf)

    model = StyleTransfer(FLAGS)
    with tf.Session(graph=model.g) as sess:
        trainer = StyleTransferTrainer(sess, model)
        trainer.train()
