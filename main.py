import argparse
import os
import jax
import wandb
import training

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--work_dir', type=str, default='/home/alishbaimran/projects/flaxmodels/training/resnet', help='Directory for logging and checkpoints.')
    parser.add_argument('--data_dir', type=str, default='/datasets/ilsvrc_2024-01-04_1601/', help='Directory for storing data.')
    parser.add_argument('--name', type=str, default='test', help='Name of this experiment.')
    parser.add_argument('--group', type=str, default='default', help='Group name of this experiment.')
    # Training
    parser.add_argument('--arch', type=str, default='resnet101', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Architecture.')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint.')
    parser.add_argument('--num_epochs', type=int, default=90, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=9, help='Number of warmup epochs with lower learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&bBiases.')
    parser.add_argument('--log_every', type=int, default=100, help='Log every log_every steps.')
    args = parser.parse_args()
    
    if jax.process_index() == 0:
        args.ckpt_dir = os.path.join(args.work_dir, args.group, args.name, 'checkpoints')
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if args.wandb:
            wandb.login(key="-")
            wandb.init(project="unet-foveal", name=args.name)

    training.train_and_evaluate(args)


if __name__ == '__main__':
    main()
