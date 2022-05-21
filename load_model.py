import torch
from torchvision.models import (
    alexnet,
    squeezenet1_0,
    squeezenet1_1,

    googlenet,
    inception_v3,

    wide_resnet50_2,
    wide_resnet101_2,

    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,

    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,

    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,

    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
    vgg11,
    vgg13,
    vgg16,
    vgg19,

    densenet121,
    densenet169,
    densenet161,
    densenet201,

    efficientnet_b5,

    resnext50_32x4d,
    resnext101_32x8d,

    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,

    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_1_6gf,
    regnet_y_3_2gf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
    
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_1_6gf,
    regnet_x_3_2gf,
    regnet_x_8gf,
    regnet_x_16gf,
    regnet_x_32gf,

    mnasnet0_5,
    mnasnet1_0,

    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,

    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large
)
from resnest.torch import resnest50, resnest101


RESIDUAL_ARCHS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'wide_resnet50_2', 'wide_resnet101_2', 'resnest50', 'resnest101',
    'resnext50_32x4d', 'resnext101_32x8d',
    'resnet50_ibn_a', 'resnet101_ibn_a', 'resnext101_ibn_a', 'se_resnet101_ibn_a'
]
ONELAYER_ARCHS = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'vgg11', 'vgg13', 'vgg16', 'vgg19'
]
DENSE_ARCHS = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]
REGNET_ARCHS = [
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf',
    'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',
    'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf',
    'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf'
]
EFFICIENT_ARCHS = [
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7'
]
LIGHT_ARCHS = [
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
]
TRANSFORMERS = [
    'vit_b_16'
]

def load_model(arch):

    # alexnet
    if arch == 'alexnet':
        model = alexnet(pretrained=True, progress=True)
        t_d = 256
        d = 100

    # SqueezeNet
    elif arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=True, progress=True)
        t_d = 512
        d = 100
    elif arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=True, progress=True)
        t_d = 512
        d = 100

    # MobileNet
    elif arch == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=True, progress=True)
        t_d = 64
        d = 64
    elif arch == 'mobilenet_v3_small':
        model = mobilenet_v3_small(pretrained=True, progress=True)
        t_d = 64
        d = 64
    elif arch == 'mobilenet_v3_large':
        model = mobilenet_v3_large(pretrained=True, progress=True)
        t_d = 64
        d = 64

    # Inception
    elif arch == 'googlenet':
        model = googlenet(pretrained=True, progress=True)
        t_d = 1248
        d = 550
    elif arch == 'inception_v3':
        model = inception_v3(pretrained=True, progress=True)
        t_d = 832
        d = 220

    # ResNet
    elif arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif arch == 'resnet34':
        model = resnet34(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif arch == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif arch == 'resnet101':
        model = resnet101(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif arch == 'resnet152':
        model = resnet152(pretrained=True, progress=True)
        t_d = 1792
        d = 550

    #Wide-ResNet
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif arch == 'wide_resnet101_2':
        model = wide_resnet101_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550

    #ResNext
    elif arch == 'resnext50_32x4d':
        model = resnext50_32x4d(pretrained=True)
        t_d = 1792
        d = 550
    elif arch == 'resnext101_32x8d':
        model = resnext101_32x8d(pretrained=True)
        t_d = 1792
        d = 550

    #ResNest
    elif arch == 'resnest50':
        model = resnest50(pretrained=True)
        t_d = 1792
        d = 550
    elif arch == 'resnest101':
        model = resnest101(pretrained=True)
        t_d = 1792
        d = 550

    #IBN-a
    elif arch == 'resnet50_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        t_d = 1792
        d = 550
    elif arch == 'resnet101_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        t_d = 1792
        d = 550
    elif arch == 'resnext101_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=True)
        t_d = 1792
        d = 550
    elif arch == 'se_resnet101_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=True)
        t_d = 1792
        d = 550

    #EfficientNet
    elif arch == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=True, progress=True)
        t_d = 80
        d = 80
    elif arch == 'efficientnet_b1':
        model = efficientnet_b1(pretrained=True, progress=True)
        t_d = 80
        d = 80
    elif arch == 'efficientnet_b2':
        model = efficientnet_b2(pretrained=True, progress=True)
        t_d = 88
        d = 88
    elif arch == 'efficientnet_b3':
        model = efficientnet_b3(pretrained=True, progress=True)
        t_d = 104
        d = 100
    elif arch == 'efficientnet_b4':
        model = efficientnet_b4(pretrained=True, progress=True)
        t_d = 112
        d = 100
    elif arch == 'efficientnet_b5':
        model = efficientnet_b5(pretrained=True, progress=True)
        t_d = 128
        d = 100
    elif arch == 'efficientnet_b6':
        model = efficientnet_b6(pretrained=True, progress=True)
        t_d = 144
        d = 100
    elif arch == 'efficientnet_b7':
        model = efficientnet_b7(pretrained=True, progress=True)
        t_d = 160
        d = 100

    # ConvNeXt
    elif arch == 'convnext_tiny':
        model = convnext_tiny(pretrained=True, progress=True)
        t_d = 480
        d = 100
    elif arch == 'convnext_small':
        model = convnext_small(pretrained=True, progress=True)
        t_d = 480
        d = 100
    elif arch == 'convnext_base':
        model = convnext_base(pretrained=True, progress=True)
        t_d = 640
        d = 200
    elif arch == 'convnext_large':
        model = convnext_large(pretrained=True, progress=True)
        t_d = 960
        d = 300

    # MnasNet
    elif arch == 'mnasnet0_5':
        model = mnasnet0_5(pretrained=True, progress=True)
        t_d = 80
        d = 80
    elif arch == 'mnasnet1_0':
        model = mnasnet1_0(pretrained=True, progress=True)
        t_d = 144
        d = 144

    # VGG
    elif arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg11':
        model = vgg11(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg13':
        model = vgg13(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg16':
        model = vgg16(pretrained=True, progress=True)
        t_d = 512
        d = 120
    elif arch == 'vgg19':
        model = vgg19(pretrained=True, progress=True)
        t_d = 512
        d = 120
    
    # DenseNet
    elif arch == 'densenet121':
        model = densenet121(pretrained=True, progress=True)
        t_d = 96
        d = 96
    elif arch == 'densenet169':
        model = densenet169(pretrained=True, progress=True)
        t_d = 96
        d = 96
    elif arch == 'densenet161':
        model = densenet161(pretrained=True, progress=True)
        t_d = 144
        d = 100
    elif arch == 'densenet201':
        model = densenet201(pretrained=True, progress=True)
        t_d = 96
        d = 96

    # ShuffleNet
    elif arch == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(pretrained=True, progress=True)
        t_d = 336
        d = 100
    elif arch == 'shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(pretrained=True, progress=True)
        t_d = 812
        d = 220

    # RegNet
    elif arch == 'regnet_y_400mf':
        model = regnet_y_400mf(pretrained=True, progress=True)
        t_d = 360
        d = 100
    elif arch == 'regnet_y_800mf':
        model = regnet_y_800mf(pretrained=True, progress=True)
        t_d = 528
        d = 150
    elif arch == 'regnet_y_1_6gf':
        model = regnet_y_1_6gf(pretrained=True, progress=True)
        t_d = 504
        d = 150
    elif arch == 'regnet_y_3_2gf':
        model = regnet_y_3_2gf(pretrained=True, progress=True)
        t_d = 864
        d = 220
    elif arch == 'regnet_y_8gf':
        model = regnet_y_8gf(pretrained=True, progress=True)
        t_d = 1568
        d = 450
    elif arch == 'regnet_y_16gf':
        model = regnet_y_16gf(pretrained=True, progress=True)
        t_d = 1904
        d = 600
    elif arch == 'regnet_y_32gf':
        model = regnet_y_32gf(pretrained=True, progress=True)
        t_d = 2320
        d = 700

    elif arch == 'regnet_x_400mf':
        model = regnet_x_400mf(pretrained=True, progress=True)
        t_d = 256
        d = 100
    elif arch == 'regnet_x_800mf':
        model = regnet_x_800mf(pretrained=True, progress=True)
        t_d = 480
        d = 100
    elif arch == 'regnet_x_1_6gf':
        model = regnet_x_1_6gf(pretrained=True, progress=True)
        t_d = 648
        d = 150
    elif arch == 'regnet_x_3_2gf':
        model = regnet_x_3_2gf(pretrained=True, progress=True)
        t_d = 720
        d = 200
    elif arch == 'regnet_x_8gf':
        model = regnet_x_8gf(pretrained=True, progress=True)
        t_d = 1040
        d = 300
    elif arch == 'regnet_x_16gf':
        model = regnet_x_16gf(pretrained=True, progress=True)
        t_d = 1664
        d = 500
    elif arch == 'regnet_x_32gf':
        model = regnet_x_32gf(pretrained=True, progress=True)
        t_d = 2352
        d = 700

    # ViT
    elif arch == 'vit_b_16':
        model = vit_b_16(pretrained=True, progress=True)
        t_d = 591
        d = 100
    elif arch == 'vit_b_32':
        model = vit_b_32(pretrained=True, progress=True)
        t_d = 150
        d = 100
    elif arch == 'vit_l_16':
        model = vit_l_16(pretrained=True, progress=True)
        t_d = 591
        d = 100
    elif arch == 'vit_l_32':
        model = vit_l_32(pretrained=True, progress=True)
        t_d = 150
        d = 100

    else:
        raise

    return model, t_d, d


def per_layer_hook(arch, model, hook):
    if arch in RESIDUAL_ARCHS:
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

    elif arch in EFFICIENT_ARCHS:
        model.features[1][-1].register_forward_hook(hook)
        model.features[2][-1].register_forward_hook(hook)
        model.features[3][-1].register_forward_hook(hook)

    elif arch in LIGHT_ARCHS:
        model.features[1].register_forward_hook(hook)
        model.features[2].register_forward_hook(hook)
        model.features[3].register_forward_hook(hook)

    elif arch == 'googlenet':
        model.inception3a.register_forward_hook(hook)
        model.inception3b.register_forward_hook(hook)
        model.inception4a.register_forward_hook(hook)

    elif arch == 'inception_v3':
        model.Mixed_5b.register_forward_hook(hook)
        model.Mixed_5c.register_forward_hook(hook)
        model.Mixed_5d.register_forward_hook(hook)

    elif arch in ONELAYER_ARCHS:
        model.features.register_forward_hook(hook)

    elif arch in DENSE_ARCHS:
        model.features.denseblock1.denselayer6.register_forward_hook(hook)
        model.features.denseblock2.denselayer12.register_forward_hook(hook)
        if arch == 'densenet121':
            model.features.denseblock3.denselayer24.register_forward_hook(hook)
        elif arch == 'densenet169':
            model.features.denseblock3.denselayer32.register_forward_hook(hook)
        elif arch == 'densenet161':
            model.features.denseblock3.denselayer36.register_forward_hook(hook)
        elif arch == 'densenet201':
            model.features.denseblock3.denselayer48.register_forward_hook(hook)

    elif arch in ['mnasnet0_5', 'mnasnet1_0']:
        model.layers[8].register_forward_hook(hook)
        model.layers[9].register_forward_hook(hook)
        model.layers[10].register_forward_hook(hook)

    elif arch in ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']:
        model.stage2[-1].register_forward_hook(hook)
        model.stage3[-1].register_forward_hook(hook)
        model.stage4[-1].register_forward_hook(hook)

    elif arch in REGNET_ARCHS:
        model.trunk_output.block1[-1].register_forward_hook(hook)
        model.trunk_output.block2[-1].register_forward_hook(hook)
        model.trunk_output.block3[-1].register_forward_hook(hook)

    elif arch in TRANSFORMERS:
        model.encoder.layers.encoder_layer_0.register_forward_hook(hook)
        model.encoder.layers.encoder_layer_1.register_forward_hook(hook)
        model.encoder.layers.encoder_layer_2.register_forward_hook(hook)
    else:
        print('BACKBONE INIT ERROR')
        return -1
    return 1