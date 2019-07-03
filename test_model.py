import chainer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from chainer import serializers, Variable
from model import Generator, Discriminator
from train import get_mnist_num, get_arguments

model_path = Path("model")
image_path = Path("images")

if not model_path.exists():
    model_path.mkdir()

if not image_path.exists():
    image_path.mkdir()


def save_grid_images(images, file_name="grid_images.png"):
    gs = GridSpec(3, 3)
    gs.update(wspace=0.1, hspace=0.1)
    for idx, image in enumerate(images):
        plt.subplot(gs[idx])
        plt.imshow(image[0], cmap='gray')
        plt.axis('off')
    plt.savefig(file_name)


def predict(size=9, pos_labels=None, neg_labels=None):
    if pos_labels is None:
        pos_labels = [1]
    if neg_labels is None:
        neg_labels = [2, 5, 7, 9]

    positive_images = get_mnist_num(pos_labels, False)[:size]
    negative_images = get_mnist_num(neg_labels, False)[:size]
    save_grid_images(positive_images, image_path / "positive_images_before.png")
    save_grid_images(negative_images, image_path / "negative_images_before.png")
    print(positive_images.shape)
    print(negative_images.shape)
    positive_images = Variable(positive_images)
    negative_images = Variable(negative_images)

    if chainer.config.user_gpu_mode:
        positive_images.to_gpu()
        negative_images.to_gpu()

    gen_model_path = model_path / "gen_iter_100000.model"
    generator = Generator()
    if chainer.config.user_gpu_mode:
        generator.to_gpu()
    serializers.load_npz(str(gen_model_path), generator)

    with chainer.using_config("train", False):
        pos_img_recon = generator(positive_images).array
        neg_img_recon = generator(negative_images).array

    if chainer.config.user_gpu_mode:
        pos_img_recon = generator.xp.asnumpy(pos_img_recon)
        neg_img_recon = generator.xp.asnumpy(neg_img_recon)

    pos_img_recon = (pos_img_recon * 255).astype(np.uint8)
    neg_img_recon = (neg_img_recon * 255).astype(np.uint8)
    save_grid_images(pos_img_recon, image_path / "positive_images_after.png")
    save_grid_images(neg_img_recon, image_path / "negative_images_after.png")


if __name__ == '__main__':
    arguments = get_arguments()
    chainer.config.user_gpu_mode = (arguments.gpu_id >= 0)
    predict()
