# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load Aircraft dataset.

This file is modified from:
    https://github.com/HaoMood/blinear-cnn-faster.
"""


import os
import pickle

import PIL.Image
import numpy as np
import torch
import torch.utils.data

__all__ = ['Aircraft']
__author__ = 'Yuan Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2019-02-22'
__email__ = 'ewyuanzhang@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2019-02-26'
__version__ = '1.0'


class Aircraft(torch.utils.data.Dataset):
    """Aircraft dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        elif download:
            url = ('http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/'
                   'fgvc-aircraft-2013b.tar.gz')
            self._download(url)
            self._extract()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'processed', 'train.pkl'), 'rb'))
            #assert (len(self._train_data) == 6667
            #        and len(self._train_labels) == 6667)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'processed', 'test.pkl'), 'rb'))
            #assert (len(self._test_data) == 3333
            #        and len(self._test_labels) == 3333)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return image, target

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
            os.path.isfile(os.path.join(self._root, 'processed', 'train.pkl'))
            and os.path.isfile(os.path.join(self._root, 'processed', 'test.pkl')))

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0x775)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw', 'fgvc-aircraft-2013b.tar.gz')
        try:
            print('Downloading ' + url + ' to ' + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, 'r:gz')
        os.chdir(os.path.join(self._root, 'raw'))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'raw/fgvc-aircraft-2013b/data/images')
        
        # Format of train_test_split.txt: <7_digit_image_id> <label>
        def txt2array(fname):
            data = []
            with open(fname, 'r') as f:
                for line in f:
                    data.append([line[:7], line[8:-1]])
            data = np.array(data)
            return data
        id2train = txt2array(os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/data/images_variant_trainval.txt'))
        id2test = txt2array(os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/data/images_variant_test.txt'))
        
        # Format of variants.txt: <label_name>
        def enumerate_label(fname):
            label2id = {}
            with open(fname, 'r') as f:
                for line in f:
                    label = line[:-1]
                    label2id[label] = len(label2id)
            label2id = label2id
            return label2id
        label2id = enumerate_label(os.path.join(
            self._root, 'raw/fgvc-aircraft-2013b/data/variants.txt'))
        
        def load_data(img_label):
            new_edge = 448
            data, labels = [], []
            for id_, label in img_label:
                image = PIL.Image.open(os.path.join(image_path, id_)+".jpg")

                # Convert gray scale image to RGB image.
                if image.getbands()[0] == 'L':
                    image = image.convert('RGB')
                if image.size[0] != new_edge and image.size[1] != new_edge:
                    if image.size[0] > image.size[1]:
                        new_size = (int(float(image.size[0]) / image.size[1] * new_edge), new_edge)
                    else:
                        new_size = (new_edge, int(float(image.size[1]) / image.size[0] * new_edge))
                    image = image.resize(new_size, PIL.Image.ANTIALIAS)
                image_np = np.array(image)
                image.close()
                
                data.append(image_np)
                labels.append(label2id[label])
            return data, labels
            
        train_data, train_labels = load_data(id2train)
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self._root, 'processed/train.pkl'), 'wb'))
        del train_data, train_labels
        test_data, test_labels = load_data(id2test)
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self._root, 'processed/test.pkl'), 'wb'))
