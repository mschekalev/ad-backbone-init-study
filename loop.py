import os
import argparse
from datetime import datetime, timedelta

# arches = [
#     'alexnet', 'googlenet', 'inception_v3',
#     'wide_resnet50_2', 'wide_resnet101_2',
#     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#     'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
#     'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
#     'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
#     'resnext50_32x4d', 'resnext101_32x8d',
#     'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
#     'regnet_y_16gf', 'regnet_y_32gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf',
#     'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf',
#     'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
# ]

arches = [
    'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'vgg13', 'vgg16', 'vgg19'
]

def main(args):
    total = len(arches)

    print('')
    print(f'EVALUATION TYPE : {args.eval}')
    print('')
    print(f'TOTAL : {total}')

    for i, arch in enumerate(arches):
        dt = datetime.strftime(datetime.utcnow() + timedelta(hours=3), '%Y-%m-%d %H:%M:%S')
        print('------------------------------------------------')
        print(f'NUM: {i + 1} / {total}')
        print('')
        print(f'ARCH: {arch}')
        print(f'Execution start: {dt}')
        print('')

        os.system(f'python main.py --arch {arch} --eval {args.eval}')

        dt = datetime.strftime(datetime.utcnow() + timedelta(hours=3), '%Y-%m-%d %H:%M:%S')
        print('')
        print(f'Execution end: {dt}')
        print('------------------------------------------------')
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluatiuon loop')
    parser.add_argument('--eval', type=str, choices=['padim', 'spade', 'both'], default='padim')
    args = parser.parse_args()

    main(args)