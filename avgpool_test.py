import load_model


arches = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'googlenet', 'inception_v3',
    'wide_resnet50_2', 'wide_resnet101_2', 'mobilenet_v2', 'mobilenet_v3_small',
    'mobilenet_v3_large', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'vgg11', 'vgg13', 'vgg16',
    'vgg19', 'densenet121', 'densenet169', 'densenet161', 'densenet201', 'efficientnet_b5',
    'resnext50_32x4d', 'resnext101_32x8d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
    'regnet_y_16gf', 'regnet_y_32gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf',
    'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf', 'mnasnet0_5',
    'mnasnet1_0', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
]


for arch in arches:
    model, _, _ = load_model.load_model(arch)

    try:
        a = model.avgpool
    except:
        print(arch)