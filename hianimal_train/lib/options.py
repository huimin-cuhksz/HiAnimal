import argparse


class BaseOptions:
    def parse(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        data = parser.add_argument_group("Data")
        data.add_argument("--dataroot", default="./data")
        data.add_argument("--loadSize", type=int, default=512)
        data.add_argument("--num_views", type=int, default=1)
        data.add_argument("--num_sample_inout", type=int, default=16000)

        training = parser.add_argument_group("Training")
        training.add_argument("--name", default="example")
        training.add_argument("--gpu_id", type=int, default=0)
        training.add_argument("--num_threads", type=int, default=1)
        training.add_argument("--serial_batches", action="store_true")
        training.add_argument("--pin_memory", action="store_true")
        training.add_argument("--batch_size", type=int, default=2)
        training.add_argument("--learning_rate", type=float, default=1e-3)
        training.add_argument("--num_epoch", type=int, default=2000)
        training.add_argument("--freq_plot", type=int, default=10)
        training.add_argument("--freq_save", type=int, default=10)
        training.add_argument("--resume_epoch", type=int, default=-1)
        training.add_argument("--continue_train", action="store_true")
        training.add_argument("--schedule", type=int, nargs="+", default=[30, 60, 90, 120])
        training.add_argument("--gamma", type=float, default=0.1)

        model = parser.add_argument_group("Model")
        model.add_argument("--z_size", type=float, default=200.0)
        model.add_argument(
            "--mlp_dim", type=int, nargs="+", default=[257, 1024, 512, 256, 128, 3]
        )
        model.add_argument("--no_residual", action="store_true")

        augmentation = parser.add_argument_group("Augmentation")
        augmentation.add_argument("--aug_bri", type=float, default=0.0)
        augmentation.add_argument("--aug_con", type=float, default=0.0)
        augmentation.add_argument("--aug_sat", type=float, default=0.0)
        augmentation.add_argument("--aug_hue", type=float, default=0.0)
        augmentation.add_argument("--aug_blur", type=float, default=0.0)

        paths = parser.add_argument_group("Paths")
        paths.add_argument("--checkpoints_path", default="./checkpoints")
        paths.add_argument("--results_path", default="./results")
        paths.add_argument("--load_netG_checkpoint_path", default=None)

        testing = parser.add_argument_group("Testing")
        testing.add_argument("--resolution", type=int, default=512)
        testing.add_argument("--num_test", type=int, default=2000)

        # Kept for log/checkpoint compatibility with the original experiment.
        parser.add_argument("--sigma", type=float, default=0.05)
        return parser.parse_args()
