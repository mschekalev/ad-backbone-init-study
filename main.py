import argparse
import torch
import matplotlib
from padim import padim
from spade import spade


def main(args):
    # device setup
    use_cuda = torch.cuda.is_available()
    device = 'cpu'  # torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')

    matplotlib.use('Agg')

    if args.eval == 'padim':
        print('Running PaDiM:')
        padim(device, use_cuda, args.arch, args.to_plot, args.to_dump)
    elif args.eval == 'spade':
        print('Running SPADE:')
        spade(device, use_cuda, args.arch, args.top_k, args.to_plot, args.to_dump)
    elif args.eval == 'both':
        print('Running PaDiM:')
        padim(device, use_cuda, args.arch, args.to_plot, args.to_dump)

        print('Running SPADE:')
        spade(device, use_cuda, args.arch, args.top_k, args.to_plot, args.to_dump)
    else:
        print('Unexpected eval argument')
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--eval', type=str, choices=['padim', 'spade', 'both'], default='padim')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--to_plot', type=int, choices=[0, 1], default=0)
    parser.add_argument('--to_dump', type=int, choices=[0, 1], default=0)
    args =  parser.parse_args()

    main(args)