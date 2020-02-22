import os
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ************************************************************
    # Branches Related
    # ************************************************************
    parser.add_argument('--compatibility', type=bool, default=False)
    parser.add_argument('--branches', nargs='+', type=str, default=['global', 'abd']) # global abd
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--global-max-pooling', type=bool, default=False)
    parser.add_argument('--global-dim', type=int, default=1024)

    parser.add_argument('--abd-dim', type=int, default=1024)
    parser.add_argument('--abd-np', type=int, default=2)
    parser.add_argument('--abd-dan', nargs='+', type=str, default=[]) # cam pam
    parser.add_argument('--abd-dan-no-head', type=bool, default=True)
    parser.add_argument('--shallow-cam', type=bool, default=True)

    # ************************************************************
    # OF Related
    # ************************************************************
    parser.add_argument('--use-of', type=bool, default=False)
    parser.add_argument('--of-beta', type=float, default=1e-6)
    parser.add_argument('--of-start-epoch', type=int, default=23)
    parser.add_argument('--of-position', nargs='+', type=str, default=['before', 'after', 'cam', 'pam', 'intermediate'])

    # ************************************************************
    # OW Related
    # ************************************************************
    parser.add_argument('--use-ow', type=bool, default=False)
    parser.add_argument('--ow-beta', type=float, default=1e-3)

    return parser


def model_kwargs(parsed_args):
    return {
        'branches': parsed_args.branches,
        'global_max_pooling': parsed_args.global_max_pooling,
        'global_dim': parsed_args.global_dim,
        'dropout': parsed_args.dropout,
        'abd_dim': parsed_args.abd_dim,
        'abd_np': parsed_args.abd_np,
        'abd_dan': parsed_args.abd_dan,
        'abd_dan_no_head': parsed_args.abd_dan_no_head,
        'shallow_cam': parsed_args.shallow_cam,
        'compatibility': parsed_args.compatibility
    }


def of_kwargs(parsed_args):
    return {
        'of_position': parsed_args.of_position,
        'of_beta': parsed_args.of_beta,
        'of_start_epoch': parsed_args.of_start_epoch
    }


def ow_kwargs(parsed_args):
    return {
        'use_ow':parsed_args.use_ow,
        'ow_beta': parsed_args.ow_beta
    }
