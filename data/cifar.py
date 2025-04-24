import torch
from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import clip

from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root

class CustomCIFARBase:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))
        self._preprocess_text_prompts()
        
    def _preprocess_text_prompts(self):
        """Pre-tokenize and cache text prompts for all classes"""
        self.text_templates = [
            "a photo of a {}",
            "a cropped photo of a {}", 
            "a good photo of a {}",
            "a bad photo of a {}",
            "a photo of one {}",
            "a photo of many {}",
            "a {} in natural scene",
            "a {} on white background",
            "a hard to see photo of a {}",
            "a low resolution photo of a {}"
        ]
        
        # Pre-tokenize all class prompts
        self.tokenized_prompts = {}
        for class_idx, class_name in enumerate(self.class_names):
            prompts = [template.format(class_name) for template in self.text_templates]
            tokenized = torch.stack([clip.tokenize(p) for p in prompts])
            self.tokenized_prompts[class_idx] = tokenized

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        text_inputs = self.tokenized_prompts[label]
        uq_idx = self.uq_idxs[item]
        return img, label, text_inputs, uq_idx

    def __len__(self):
        return len(self.targets)

class CustomCIFAR10(CustomCIFARBase, CIFAR10):
    def __init__(self, *args, **kwargs):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        super().__init__(*args, **kwargs)


class CustomCIFAR100(CustomCIFARBase, CIFAR100):
    def __init__(self, *args, **kwargs):
        # Use fine-grained CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        super().__init__(*args, **kwargs)


class TextPromptCIFAR(CIFAR100):
    def __init__(self, base_dataset, text_prompts=None):
        self.data = base_dataset.data
        self.targets = base_dataset.targets
        self.transform = base_dataset.transform
        self.uq_idxs = base_dataset.uq_idxs
        self.class_names = base_dataset.class_names
        self.text_prompts = text_prompts or [
            "a photo of a {}",
            "a cropped photo of a {}",
            "a good photo of a {}",
            "a bad photo of a {}",
            "a photo of one {}",
            "a photo of many {}"
        ]
        
    def __getitem__(self, index):
        image, label, uq_idx = super().__getitem__(index)
        class_name = self.class_names[label]
        text_inputs = [clip.tokenize(prompt.format(class_name))[0] for prompt in self.text_prompts]
        return image, label, text_inputs, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):
    if len(idxs) > 0:
        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]
        return dataset
    else:
        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]
        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)
    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0):
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True, download=True)

    # Get labelled training set
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False, download=True)

    # Wrap datasets with TextPromptCIFAR
    train_dataset_labelled = TextPromptCIFAR(train_dataset_labelled_split if split_train_val else train_dataset_labelled)
    train_dataset_unlabelled = TextPromptCIFAR(train_dataset_unlabelled)
    val_dataset_labelled = TextPromptCIFAR(val_dataset_labelled_split) if split_train_val else None
    test_dataset = TextPromptCIFAR(test_dataset)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0):
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=cifar_100_root, 
                                      transform=train_transform, 
                                      train=True, 
                                      download=True)

    # Get labelled training set
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), 
                                          include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, 
                                         prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), 
                                              np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=cifar_100_root, 
                                transform=test_transform, 
                                train=False, 
                                download=True)

    all_datasets = {
        'train_labelled': train_dataset_labelled_split if split_train_val else train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled_split if split_train_val else None,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':
    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
    
