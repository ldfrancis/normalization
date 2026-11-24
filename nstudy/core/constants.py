"""Fixed values shared across the codebase."""

# Per-channel mean and standard deviation of the training images, for inputs in [0, 1].
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR_TRAIN_SIZE = 50_000

# Special tokens of the translation vocabulary. Their ids are their positions here.
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<s>", "</s>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
