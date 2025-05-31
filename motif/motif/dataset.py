import json
import os
import random
import time
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import torch
import tqdm


class AnnotationIdx:
    FIRST = 0
    SECOND = 1
    TIE = 2
    UNKOWN = 3


class InfiniteDataLoader:
    def __init__(self, dataloader_fn):
        self.dataloader_fn = dataloader_fn
        self.loader = iter(self.dataloader_fn())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader_fn())
            return next(self.loader)


def dict_collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        # stack along a new dimension
        if isinstance(batch[0][key], torch.Tensor):
            stacked_tensor = torch.stack([item[key] for item in batch])
        else:
            stacked_tensor = np.stack([item[key] for item in batch])
        collated_batch[key] = stacked_tensor
    return collated_batch


def tuple_dict_collate_fn(batch):
    use_tensor = isinstance(batch[0][1][list(batch[0][1].keys())[0]], torch.Tensor)
    collated_data = {}
    keys = batch[0][0].keys()
    for key in keys:
        # stack along a new dimension
        if use_tensor:
            stacked_tensor = torch.stack([item[0][key] for item in batch])
        else:
            stacked_tensor = np.stack([item[0][key] for item in batch])
        collated_data[key] = stacked_tensor
    collated_infos = {}
    keys = batch[0][1].keys()
    for key in keys:
        # stack along a new dimension
        if use_tensor:
            stacked_tensor = torch.stack([item[1][key] for item in batch])
        else:
            stacked_tensor = np.stack([item[1][key] for item in batch])
        collated_infos[key] = stacked_tensor
    return collated_data, collated_infos


def flatten_pair_collate_fn(batch, ignore_keys: List[str] = []):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        # stack along a new dimension
        if key not in ignore_keys:
            # Batch size, 2, sequence length, ...
            stacked_tensor = torch.stack([item[key] for item in batch])
            collapsed_tensor = stacked_tensor.flatten(0, 2)
            collated_batch[key] = collapsed_tensor
        else:
            collated_batch[key] = [el[key] for el in batch]
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key])
            else:
                collated_batch[key] = np.stack(collated_batch[key])
    return collated_batch


def pair_collate_fn(batch):
    # `batch` is a list of subepisode dictionaries
    # Group them into pairs
    pairs = [(batch[i], batch[i + 1]) for i in range(0, len(batch), 2)]

    # Turn each pair of dictionaries into a dictionary of tensors (batch_size, 2, subepisode_length, dims)
    collated_pairs = {key: np.array([(pair[0][0][key], pair[1][0][key]) for pair in pairs]) for key in pairs[0][0][0]}

    collated_infos = {
        key: (
            np.array([((pair[0][1][key]), pair[1][1][key]) for pair in pairs])
            if type(pairs[0][0][1][key]) != str
            else [(pair[0][1][key], pair[1][1][key]) for pair in pairs]
        )
        for key in pairs[0][0][1]
    }
    return collated_pairs, collated_infos


class PairsDataset(torch.utils.data.Dataset):
    """Dataset for pairs of subepisodes.

    Args:
        directory (str): Path to the directory containing the subepisodes.
        data_keys (Optional[List[str]]): List of keys to extract from the subepisodes.
        info_keys (Optional[List[str]]): List of keys to extract from the info.
        preference_keys (Optional[List[str]]): List of keys to extract from the preferences.
        transform (Optional[Callable]): Transform to apply to the data.
        mode (str): Mode to open the subepisode files in. See `numpy.load` for details.
    """

    def __init__(
        self,
        directory: str,
        data_keys: Optional[List[str]] = None,
        info_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        preference_keys: Optional[List[str]] = None,
        mode: str = "r",
    ):
        assert data_keys is not None or info_keys is not None or preference_keys is not None
        self.directory = directory
        self.data_keys = data_keys
        self.info_keys = info_keys
        self.preference_keys = preference_keys
        # Use directory structure to load arrays with the right keys
        if self.data_keys is not None:
            self.data_arrays = self.load_to_dict(self.directory, "data", mode, data_keys)
            self.valid_indices = np.arange(len(self.data_arrays[list(self.data_arrays.keys())[0]]))
        if self.info_keys is not None:
            self.info_arrays = self.load_to_dict(self.directory, "info", mode, info_keys)
            self.valid_indices = np.arange(len(self.info_arrays[list(self.info_arrays.keys())[0]]))
        final_mask = np.ones(self.data_arrays[list(self.data_arrays.keys())[0]].shape[0], dtype=bool)
        if self.preference_keys is not None:
            self.pref_arrays = self.load_to_dict(self.directory, "preference", mode, preference_keys)
            print("loaded preference array from: ", os.path.join(self.directory, "preference", preference_keys[0]))
            mask_arrays = {key: self.pref_arrays[key] != AnnotationIdx.UNKOWN for key in self.pref_arrays}
            for _, arr in mask_arrays.items():
                print(f'mask arrays shape: {arr.shape}')
            for _, arr in mask_arrays.items():
                final_mask = np.logical_and(final_mask, arr)
        self.valid_indices = np.where(final_mask)[0]

        self.transform = transform

    def load_to_dict(self, dir_name: str, subdir_name: str, mode: str = "r", keys: Optional[List[str]] = None):
        running_keys = set(keys)
        array_dict = {}
        for filename in os.listdir(os.path.join(self.directory, subdir_name)):
            if keys is None or filename.split(".")[0] in keys:
                if "images" in filename.split(".")[0]:
                    import pickle
                    print("Starting to load image data!")
                    with open(os.path.join(dir_name, subdir_name, filename), "rb") as data:
                        array = pickle.load(data)
                    # array = hickle.load(os.path.join(dir_name, subdir_name, filename))
                    if array.shape[-1] == 3:
                        array = array.transpose(tuple(range(len(array.shape[:-3]))) + (-1, -3, -2))
                    print("Finished loading image data!")
                else:
                    array = np.load(os.path.join(dir_name, subdir_name, filename), mmap_mode=mode)
                array_dict[filename.split(".")[0]] = array
                running_keys.remove(filename.split(".")[0])
        if len(running_keys) > 0:
            raise ValueError(f"Keys {running_keys} not found in {os.path.join(dir_name,subdir_name)}")
        return array_dict

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        data = {"idx": idx}
        if self.data_keys is not None:
            for key in self.data_arrays:
                data[key] = self.data_arrays[key][self.valid_indices[idx]]
        if self.info_keys is not None:
            for key in self.info_arrays:
                data[key] = self.info_arrays[key][self.valid_indices[idx]]
        if self.preference_keys is not None:
            for key in self.pref_arrays:
                data[key + "_pref"] = self.pref_arrays[key][self.valid_indices[idx]]
        if self.transform:
            data = self.transform(data)

        return data


if __name__ == "__main__":
    from torch.utils.data import random_split
    from torchvision import transforms
    from utils.preprocessing import GPT5BaselineTransform, ToTensorDict

    # info_keys = None
    # print("Loading dataset...")
    # dataset = PairsDataset(
    #     # "/media/csancaktar/Elements/motif_dataset",
    #     "/is/cluster/fast/csancaktar/motif_dataset",
    #     # data_keys=["images"],
    #     data_keys=["clip_embedding_cam_right", "observations"],
    #     info_keys=info_keys,
    #     preference_keys=["preferences_gt"],
    #     transform=transforms.Compose([ToTensorDict()]),
    #     mode="r",
    # )

    # print(dir(dataset))

    # print("Dataset length", len(dataset))
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(
    #     dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    # )
    # # print(dir(train_dataset))
    # # print(len(train_dataset.indices))
    # # # print(train_dataset.data_arrays["clip_embedding_cam_right"].shape)
    # # test = {k: v[train_dataset.indices, ...].reshape(-1, v.shape[-1]) for k, v in dataset.data_arrays.items()}

    # # for key in test.keys():
    # #     print(f"Key : {key} and shape {test[key].shape}")
    # # exit()

    # # data_loader = torch.utils.data.DataLoader(
    # #     train_dataset, batch_size=64, shuffle=True, collate_fn=dict_collate_fn, num_workers=1
    # # )

    # collate_ignore_keys = ["idx", "preferences_gt_pref"]
    # data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     collate_fn=partial(flatten_pair_collate_fn, ignore_keys=collate_ignore_keys),
    #     num_workers=10,
    # )

    # start_time = time.time()
    # for batch in tqdm.tqdm(data_loader):
    #     pass

    # end_time = time.time()
    # epoch_time = end_time - start_time
    # print(f"Time taken for one epoch: {epoch_time} seconds")

    # for key in batch.keys():
    #     print(f"Key : {key} and shape {batch[key].shape}")


    ## TRYING NEW DATASET!!!

    info_keys = None
    print("Loading dataset...")
    dataset = PairsDataset(
        # "/media/csancaktar/Elements/motif_dataset",
        "/fast/csancaktar/sensei_datasets/minihack/keychest_random_dataset_100K",
        # data_keys=["images"],
        data_keys=["images", "observations"],
        info_keys=info_keys,
        preference_keys=["preferences_gpt4"],
        transform=transforms.Compose([ToTensorDict()]),
        mode="r",
    )

    print(dir(dataset))

    print("Dataset length", len(dataset))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    # print(dir(train_dataset))
    # print(len(train_dataset.indices))
    # # print(train_dataset.data_arrays["clip_embedding_cam_right"].shape)
    # test = {k: v[train_dataset.indices, ...].reshape(-1, v.shape[-1]) for k, v in dataset.data_arrays.items()}

    # for key in test.keys():
    #     print(f"Key : {key} and shape {test[key].shape}")
    # exit()

    # data_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=64, shuffle=True, collate_fn=dict_collate_fn, num_workers=1
    # )

    collate_ignore_keys = ["idx", "preferences_gpt4_pref"]
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=partial(flatten_pair_collate_fn, ignore_keys=collate_ignore_keys),
        num_workers=10,
    )

    start_time = time.time()
    for batch in tqdm.tqdm(data_loader):
        for key in batch.keys():
            print(f"Key : {key} and shape {batch[key].shape}")
        break

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time taken for one epoch: {epoch_time} seconds")

    for key in batch.keys():
        print(f"Key : {key} and shape {batch[key].shape}")
