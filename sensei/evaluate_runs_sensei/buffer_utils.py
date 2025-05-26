import imageio
from PIL import Image
from dreamerv3.embodied.core.path import Path
import os
import pkg_resources
import numpy as np
import pickle 

WARRIOR_ID = 337
FARMER_ID = 339
STAIRCASE_UP_ID = 2382
FLOOR_ID = 2378

def scan(directory, capacity=None, shorten=0):
    directory = Path(directory)
    filenames, total = [], 0
    for filename in sorted(directory.glob('*.npz')):
        if capacity and total >= capacity:
            break
        # print(filename)
        filenames.append(filename)
        total += max(0, int(filename.stem.split('-')[3]) - shorten)
    # print(total)
    return filenames



def collect_data(data_dict, data_dir, include_non_finished, discrete_action=False, max_data=-1):
    for key, _ in data_dict.items():
        data_dict[key] = []

    total_length = 0
    filenames = scan(data_dir, capacity=None, shorten=0)

    non_finished_uuids = []
    non_finished_uuids_filenames = []
    for filename in filenames:
        with Path(filename).open("rb") as f:
            if filename.stem.split("-")[2] == "0000000000000000000000":
                non_finished_uuids.append(filename.stem.split("-")[1])
                non_finished_uuids_filenames.append(filename)
                pass
            else:
                data = np.load(f)
                length = int(filename.stem.split("-")[3])
                # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                if max_data > 0 and total_length + length > max_data:
                    break
                total_length += length
                for key, val in data_dict.items():
                    if key == "glyphs_crop":
                        val.extend(data["glyphs_crop"][:length, ::13, ::13])
                    elif key == "action":
                        if discrete_action:
                            val.extend(np.where(data["action"][:length, ...])[0])
                        else:
                            val.extend(data["action"][:length, ...])
                    elif isinstance(val, list) or len(val.shape)==1:
                        val.extend(data[key][:length])
                    else:
                        val.extend(data[key][:length, ...])                    

                if filename.stem.split("-")[1] in non_finished_uuids:
                    ind = non_finished_uuids.index(filename.stem.split("-")[1])
                    non_finished_uuids.pop(ind)
                    non_finished_uuids_filenames.pop(ind)
    print("Length of non-finished uuids: ", len(non_finished_uuids_filenames))
    print("Total length without duplicates: ", total_length)

    if include_non_finished:
        for filename in non_finished_uuids_filenames:
            with Path(filename).open("rb") as f:
                data = np.load(f)
                length = int(filename.stem.split("-")[3])
                # first_indices = np.where(np.logical_and(np.append(np.diff(data["is_first"][:length]), 0)!=0, data["is_first"][:length]))
                total_length += length

                for key, val in data_dict.items():
                    if key == "glyphs_crop":
                        val.extend(data["glyphs_crop"][:length, ::13, ::13])
                    elif key == "action":
                        if discrete_action:
                            val.extend(np.where(data["action"][:length, ...])[0])
                        else:
                            val.extend(data["action"][:length, ...])
                    elif isinstance(val, list) or len(val.shape)==1:
                        val.extend(data[key][:length])
                    else:
                        val.extend(data[key][:length, ...])
    
    print("Total length after adding non_finished files as well: ", total_length)

    for key, val in data_dict.items():
        data_dict[key] = np.asarray(val)

    return data_dict, total_length