import torch.nn as nn
from torch.nn import init


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def init_weights(network, init_type="normal", init_gain=0.02):
    def initialize(module):
        class_name = module.__class__.__name__
        if hasattr(module, "weight") and (
            "Conv" in class_name or "Linear" in class_name
        ):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Unknown initialization: {init_type}")
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in class_name:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    network.apply(initialize)


def init_net(network, init_type="normal", init_gain=0.02):
    init_weights(network, init_type, init_gain)
    return network
