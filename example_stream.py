import argparse
import json
import matplotlib.pyplot as plt
from loading_utils import get_stream_data_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='openloris', choices=['openloris', 'stream51'])
    parser.add_argument('--images_dir', type=str, default='/media/tyler/Data/datasets/OpenLORIS')

    # iid - randomly shuffled
    # class_iid - organized by class, then shuffled within class
    # instance - randomly shuffle all object instance videos
    # class_instance - organized by class, then view videos from class
    # instance_small - only relevant for openloris. only view one random video from each class (see low_shot_instance
    # ordering from our paper: https://arxiv.org/abs/2203.10681)
    # instance_small_121 = only relevant for openloris. only view one random video from each of the 121 object instances
    parser.add_argument('--order', type=str, default='instance',
                        choices=['iid', 'class_iid', 'instance', 'class_instance', 'instance_small',
                                 'instance_small_121'])

    # only relevant for openloris. we could choose for labels to be at the class level (40 total) or the instance level
    # (121 total). I typically go for the class level
    parser.add_argument('--label_level', type=str, choices=['class', 'instance'], default='class')

    # train batch size (1 for streaming)
    parser.add_argument('--batch_size', type=int, default=1)

    # test batch size
    parser.add_argument('--test_batch_size', type=int, default=128)

    # random shuffle seed for reproducibility and multiple runs
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # when making train_loader, keep shuffle=False or else the specified orders will be ruined
    train_loader = get_stream_data_loader(args.images_dir, True, ordering=args.order, batch_size=args.batch_size,
                                          shuffle=False, augment=False, seed=args.seed,
                                          label_level=args.label_level, dataset_name=args.dataset)
    test_loader = get_stream_data_loader(args.images_dir, False, ordering=None, batch_size=args.test_batch_size,
                                         seen_classes=None, label_level=args.label_level, dataset_name=args.dataset)

    print('train dataset len ', len(train_loader.dataset))
    print('test dataset len ', len(test_loader.dataset))

    for ix, (image, label) in enumerate(train_loader):
        if ix > 5:
            break

        # show a few images for verification (note that these are mean and standard deviation normalized so they might
        # look a bit strange)
        plt.imshow(image.numpy().transpose(0, 2, 3, 1).squeeze())
        plt.show()


if __name__ == '__main__':
    main()
