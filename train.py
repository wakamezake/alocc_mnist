# coding: utf-8
import argparse
import numpy as np
import cv2
import chainer
import chainer.functions as F

from pathlib import Path
from chainer import Variable, iterators, optimizers
from chainer.training import updaters, Trainer, extensions
from chainer import datasets
from model import Generator, Discriminator, EvalModel, ExtendedClassifier


class GANUpdater(updaters.StandardUpdater):
    def __init__(self, iterator, gen_opt, dis_opt, l2_lam, noise_std, n_dis=1, **kwds):
        opts = {
            "gen": gen_opt,
            "dis": dis_opt
        }
        iters = {"main": iterator}
        self.n_dis = n_dis
        self.l2_lam = l2_lam
        self.noise_std = noise_std
        super().__init__(iters, opts, **kwds)

    def get_batch(self):
        x = self.get_iterator("main").next()
        x = np.stack(x)

        noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        x_noisy = np.clip(x + noise, 0.0, 1.0)  # ガウシアンノイズを付加

        x = Variable(x)
        x_noisy = Variable(x_noisy)

        if chainer.config.user_gpu_mode:
            x.to_gpu()
            x_noisy.to_gpu()

        return x, x_noisy

    def update_core(self):
        opt_gen = self.get_optimizer("gen")
        opt_dis = self.get_optimizer("dis")
        gen = opt_gen.target
        dis = opt_dis.target

        # update discriminator
        # 本物に対しては1，偽物に対しては0を出すように学習
        for i in range(self.n_dis):
            x, x_noisy = self.get_batch()
            x_fake = gen(x_noisy)

            d_real = dis(x)
            ones = dis.xp.ones(d_real.shape[0], dtype=np.int32)
            loss_d_real = F.softmax_cross_entropy(d_real, ones)

            d_fake = dis(x_fake)
            zeros = dis.xp.zeros(d_fake.shape[0], dtype=np.int32)
            loss_d_fake = F.softmax_cross_entropy(d_fake, zeros)

            loss_dis = loss_d_real + loss_d_fake

            dis.cleargrads()
            loss_dis.backward()
            opt_dis.update()

        # update generator
        # 生成した画像に対してDが1を出すようにする
        x, x_noisy = self.get_batch()
        x_fake = gen(x_noisy)

        d_fake = dis(x_fake)
        ones = dis.xp.ones(d_fake.shape[0], dtype=np.int32)
        loss_gen = F.softmax_cross_entropy(d_fake, ones)

        loss_gen_l2 = F.mean_squared_error(x, x_fake)

        loss_gen_total = loss_gen + self.l2_lam * loss_gen_l2

        gen.cleargrads()
        dis.cleargrads()
        loss_gen_total.backward()
        opt_gen.update()

        chainer.report({
            "generator/loss": loss_gen,
            "generator/l2": loss_gen_l2,
            "discriminator/loss": loss_dis
        })


# 生成画像を保存するextension
def ext_save_img(generator, pos_data, neg_data, out: Path, noise_std):
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        pass

    @chainer.training.make_extension()
    def _ext_save_img(trainer):
        # 画像取得
        i = np.random.randint(len(pos_data))
        pos_img = pos_data[i][0]
        i = np.random.randint(len(neg_data))
        neg_img = neg_data[i][0]

        # ノイズ付加
        noise = np.random.normal(0, noise_std, size=pos_img.shape).astype(np.float32)
        pos_img = np.clip(pos_img + noise, 0.0, 1.0)
        noise = np.random.normal(0, noise_std, size=neg_img.shape).astype(np.float32)
        neg_img = np.clip(neg_img + noise, 0.0, 1.0)

        # 保存
        temp = np.squeeze(pos_img * 255).astype(np.uint8)
        cv2.imwrite(str(out / "in_pos_iter_{:06d}.png".format(trainer.updater.iteration)), temp)
        temp = np.squeeze(neg_img * 255).astype(np.uint8)
        cv2.imwrite(str(out / "in_neg_iter_{:06d}.png".format(trainer.updater.iteration)), temp)

        # shapeを調整
        pos_img = np.expand_dims(pos_img, axis=0)
        neg_img = np.expand_dims(neg_img, axis=0)

        pos_img = Variable(pos_img)
        neg_img = Variable(neg_img)
        if chainer.config.user_gpu_mode:
            pos_img.to_gpu()
            neg_img.to_gpu()

        with chainer.using_config("train", False):
            # 再構築
            pos_recon = generator(pos_img).array
            neg_recon = generator(neg_img).array

        # 再構築画像を保存
        if chainer.config.user_gpu_mode:
            pos_recon = generator.xp.asnumpy(pos_recon)
            neg_recon = generator.xp.asnumpy(neg_recon)

        pos_recon = np.squeeze(pos_recon * 255).astype(np.uint8)
        neg_recon = np.squeeze(neg_recon * 255).astype(np.uint8)

        cv2.imwrite(str(out / "out_pos_iter_{:06d}.png".format(trainer.updater.iteration)), pos_recon)
        cv2.imwrite(str(out / "out_neg_iter_{:06d}.png".format(trainer.updater.iteration)), neg_recon)

    return _ext_save_img


def get_mnist_num(dig_list: list, train=True) -> np.ndarray:
    """
    指定した数字の画像だけ返す
    """
    mnist_dataset = datasets.get_mnist(ndim=3)[0 if train else 1]  # MNISTデータ取得
    mnist_dataset = [img for img, label in mnist_dataset[:] if label in dig_list]
    mnist_dataset = np.stack(mnist_dataset)
    return mnist_dataset


def main(arguments, neg_labels, pos_labels):
    output_dir_path = Path(arguments.output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    # settings
    adam_setting = {"alpha": arguments.adam_alpha,
                    "beta1": arguments.adam_beta1,
                    "beta2": arguments.adam_beta2}

    updater_setting = {"n_dis": arguments.n_dis,
                       "l2_lam": arguments.l2_lam,
                       "noise_std": arguments.noise_std}
    chainer.config.user_gpu_mode = (arguments.gpu_id >= 0)
    if chainer.config.user_gpu_mode:
        chainer.backends.cuda.get_device_from_id(arguments.gpu_id ).use()

    # 訓練用正常データ
    mnist_neg = get_mnist_num(neg_labels)

    # iteratorを作成
    iterator_setting = {
        "batch_size": arguments.batch_size,
        "shuffle": True,
        "repeat": True
    }
    neg_iter = iterators.SerialIterator(mnist_neg, **iterator_setting)

    generator = Generator()
    discriminator = Discriminator()
    if chainer.config.user_gpu_mode:
        generator.to_gpu()
        discriminator.to_gpu()

    opt_g = optimizers.Adam(**adam_setting)
    opt_g.setup(generator)
    opt_d = optimizers.Adam(**adam_setting)
    opt_d.setup(discriminator)
    if arguments.weight_decay > 0.0:
        opt_g.add_hook(chainer.optimizer.WeightDecay(arguments.weight_decay))
        opt_d.add_hook(chainer.optimizer.WeightDecay(arguments.weight_decay))

    updater = GANUpdater(neg_iter, opt_g, opt_d, **updater_setting)
    trainer = Trainer(updater, (arguments.iteration, "iteration"), out=arguments.result_dir)

    # テストデータを取得
    test_neg = get_mnist_num(neg_labels, train=False)
    test_pos = get_mnist_num(pos_labels, train=False)
    # 正常にラベル0，異常にラベル1を付与
    test_neg = chainer.datasets.TupleDataset(test_neg, np.zeros(len(test_neg), dtype=np.int32))
    test_pos = chainer.datasets.TupleDataset(test_pos, np.ones(len(test_pos), dtype=np.int32))
    test_ds = chainer.datasets.ConcatenatedDataset(test_neg, test_pos)
    test_iter = iterators.SerialIterator(test_ds, repeat=False, shuffle=True, batch_size=500)

    ev_target = EvalModel(generator, discriminator, arguments.noise_std)
    ev_target = ExtendedClassifier(ev_target)
    if chainer.config.user_gpu_mode:
        ev_target.to_gpu()
    evaluator = extensions.Evaluator(test_iter, ev_target, device=arguments.g if chainer.config.user_gpu_mode else None)
    trainer.extend(evaluator)

    # 訓練経過の表示などの設定
    trigger = (5000, "iteration")
    trainer.extend(extensions.LogReport(trigger=trigger))
    trainer.extend(extensions.PrintReport(["iteration", "generator/loss", "generator/l2", "discriminator/loss"]),
                   trigger=trigger)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(
        extensions.PlotReport(("generator/loss", "discriminator/loss"), "iteration", file_name="loss_plot.eps",
                              trigger=trigger))
    trainer.extend(extensions.PlotReport(["generator/l2"], "iteration", file_name="gen_l2_plot.eps", trigger=trigger))
    trainer.extend(
        extensions.PlotReport(("validation/main/F", "validation/main/accuracy"), "iteration", file_name="acc_plot.eps",
                              trigger=trigger))
    trainer.extend(
        ext_save_img(generator, test_pos, test_neg, output_dir_path / "out_images", arguments.noise_std),
        trigger=trigger)
    trainer.extend(extensions.snapshot_object(generator, "gen_iter_{.updater.iteration:06d}.model"), trigger=trigger)
    trainer.extend(extensions.snapshot_object(discriminator, "dis_iter_{.updater.iteration:06d}.model"),
                   trigger=trigger)

    # 訓練開始
    trainer.run()


def get_arguments():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--epochs", type=int, default=100)
    _parser.add_argument("--batch_size", type=int, default=128)
    _parser.add_argument("--iteration", type=int, default=100000)
    _parser.add_argument("output_dir", type=str)
    _parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID (negative value indicates CPU mode)")
    _parser.add_argument("--adam_alpha", type=float, default=0.0002)
    _parser.add_argument("--adam_beta1", type=float, default=0.5)
    _parser.add_argument("--adam_beta2", type=float, default=0.9)
    _parser.add_argument("--n_dis", type=int, default=1)
    _parser.add_argument("--l2_lam", type=float, default=0.2)
    _parser.add_argument("--noise_std", type=float, default=0.155)
    _parser.add_argument("--weight_decay", type=float, default=0.00001)
    _args = _parser.parse_args()
    return _args


if __name__ == "__main__":
    args = get_arguments()
    negative_labels = [1]
    positive_labels = [0]
    main(args, neg_labels=negative_labels, pos_labels=positive_labels)
