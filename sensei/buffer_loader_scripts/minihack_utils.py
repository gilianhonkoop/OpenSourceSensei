import itertools
import random
import numpy as np

substrings = ['key', 'door', 'staircase down']

unique_descriptions= ['floor of a room',
 'human rogue called Agent',
 'wall',
 'staircase up',
 'a key',
 '',
 'a key named The Master Key of Thievery',
 'dark part of a room',
 'closed door',
 'open door',
 'staircase down']

def decode_descriptions(description_encoded):
    description_decoded = []

    for i in range(description_encoded.shape[0]):
        for j in range(description_encoded.shape[1]):
            description_decoded.append("".join([chr(c) for c in description_encoded[i,j,:] if c >= 32]))
    return description_decoded

def check_interestingness_inventory(desc_encoded1, desc_encoded2, inv_glyphs1, inv_glpyhs2):
    desc_decoded1 = decode_descriptions(desc_encoded1)
    desc_decoded2 = decode_descriptions(desc_encoded2)

    # contains_substring1 = [sub for sub in substrings for s in desc_decoded1 if sub in s]
    # contains_substring2 = [sub for sub in substrings for s in desc_decoded2 if sub in s]

    contains_key_desc1 = ['key' for s in desc_decoded1 if 'key' in s]
    contains_key_desc2 = ['key' for s in desc_decoded2 if 'key' in s]
    
    contains_door_desc1 = ['closed door' for s in desc_decoded1 if 'closed door' in s]
    contains_door_desc2 = ['closed door' for s in desc_decoded2 if 'closed door' in s]
        
    contains_opendoor_desc1 = ['open door' for s in desc_decoded1 if 'open door' in s]
    contains_opendoor_desc2 = ['open door' for s in desc_decoded2 if 'open door' in s]

    contains_staircase_desc1 = ['staircase down' for s in desc_decoded1 if 'staircase down' in s]
    contains_staircase_desc2 = ['staircase down' for s in desc_decoded2 if 'staircase down' in s]
    
    key_glyph_id = 2102
    
    key_in_inventory1 = key_glyph_id in inv_glyphs1
    key_in_inventory2 = key_glyph_id in inv_glpyhs2

    if contains_staircase_desc1 and not contains_staircase_desc2:
        return 0
    elif not contains_staircase_desc1 and contains_staircase_desc2:
        return 1
    elif contains_opendoor_desc1 and not contains_opendoor_desc2:
        return 0
    elif not contains_opendoor_desc1 and contains_opendoor_desc2:
        return 1 # remove these first two to not differentiate between the open and closed doors!
    elif contains_opendoor_desc1 and contains_opendoor_desc2:
        if contains_staircase_desc1 and not contains_staircase_desc2:
            return 0
        if not contains_staircase_desc1 and contains_staircase_desc2:
            return 1
        else:
            return 2
    elif key_in_inventory1 and not key_in_inventory2:
        return 0
    elif not key_in_inventory1 and key_in_inventory2:
        return 1
    elif key_in_inventory1 and key_in_inventory2:
        if contains_door_desc1 and not contains_door_desc2:
            return 0
        elif not contains_door_desc1 and contains_door_desc2:
            return 1
        else:
            return 2
    elif contains_key_desc1 and not contains_key_desc2:
        return 0
    elif not contains_key_desc1 and contains_key_desc2:
        return 1
    elif contains_key_desc1 and contains_key_desc2:
        return 2
    # elif contains_door_desc1 and not contains_door_desc2:
    #     return 0
    # elif not contains_door_desc1 and contains_door_desc2:
    #     return 1
    # elif contains_door_desc2 and contains_door_desc2:
    #     return 2
    else:
        return 3


def check_interestingness_binary(desc_encoded1, desc_encoded2):
    desc_decoded1 = decode_descriptions(desc_encoded1)
    desc_decoded2 = decode_descriptions(desc_encoded2)

    contains_substring1 = any(sub for sub in substrings for s in desc_decoded1 if sub in s)
    contains_substring2 = any(sub for sub in substrings for s in desc_decoded2 if sub in s)
    
    if contains_substring1 and not contains_substring2:
        return 0
    elif not contains_substring1 and contains_substring2:
        return 1
    elif contains_substring1 and contains_substring2:
        return 2
    else:
        return 3


def check_interestingness_num_items(desc_encoded1, desc_encoded2):
    desc_decoded1 = decode_descriptions(desc_encoded1)
    desc_decoded2 = decode_descriptions(desc_encoded2)

    contains_substring1 = [sub for sub in substrings for s in desc_decoded1 if sub in s]
    contains_substring2 = [sub for sub in substrings for s in desc_decoded2 if sub in s]
    
    
    if any(contains_substring1) and not any(contains_substring2):
        return 0
    elif not any(contains_substring1) and any(contains_substring2):
        return 1
    elif any(contains_substring1) and any(contains_substring2):
        if len(contains_substring1) > len(contains_substring2):
            return 0
        elif len(contains_substring1) < len(contains_substring2):
            return 1
        else:
            return 2
    else:
        return 3


    
def find_substring_coordinates(description, substrings, grid_width=5):
    """
    Find the first occurrence of any of the substrings in the grid and return its coordinates.
    
    :param strings: List of strings representing the grid row-wise.
    :param substrings: List of substrings to search for.
    :param grid_width: The width of the grid (default is 5 for a 5x5 grid).
    :return: Tuple of (row, column) if a substring is found, otherwise None.
    """
    all_coordinates = []
    
    for index, string in enumerate(description):
        if any(sub in string for sub in substrings):
            # Calculate the row and column from the index
            row = index // grid_width
            column = index % grid_width
            all_coordinates.append((row, column))
            
    return all_coordinates


    
def check_interestingness_location(desc_encoded1, desc_encoded2):
    desc_decoded1 = decode_descriptions(desc_encoded1)
    desc_decoded2 = decode_descriptions(desc_encoded2)

    coordinates1 = find_substring_coordinates(desc_decoded1, substrings)
    coordinates2 = find_substring_coordinates(desc_decoded2, substrings)
    
    agent_loc = np.array([2,2])
    if any(coordinates1) and not any(coordinates2):
        return 0
    elif not any(coordinates1) and any(coordinates2):
        return 1
    elif any(coordinates1) and any(coordinates2):
        # compare distances!
        distance1 = 999
        for coordinate in coordinates1:
            current_dis = np.linalg.norm(np.asarray(coordinate)-agent_loc, ord=1)
            if current_dis < distance1:
                distance1 = current_dis

        distance2 = 999
        for coordinate in coordinates2:
            current_dis = np.linalg.norm(np.asarray(coordinate)-agent_loc, ord=1)
            if current_dis < distance2:
                distance2 = current_dis
        
        if distance1 < distance2:
            return 0
        elif distance1 > distance2:
            return 1
        else:
            return 2
    else:
        return 3


def generate_distant_pairs(length, min_distance, N):
    """
    Efficiently generate N unique pairs from a list of vectors ensuring that each
    pair is at least `min_distance` apart. This function is designed to handle large
    lists efficiently.

    :param vectors: List of vectors.
    :param min_distance: Minimum index distance between elements of each pair.
    :param N: Number of unique pairs to generate.
    :return: List of tuples, each containing a pair of vectors.
    """
    if min_distance >= length:
        raise ValueError("min_distance is too large for the length of the vector list.")

    # Function to generate pairs
    def pair_generator():
        seen = set()
        while True:
            i = random.randint(0, length - 1)
            # Generate j such that distance condition is met
            j_choices = list(range(0, max(0, i - min_distance))) + list(range(min(i + min_distance, length),length))
            if not j_choices:  # No valid j if j_choices is empty
                continue
            j = random.choice(j_choices)
            if (i, j) not in seen and (j, i) not in seen:
                seen.add((i, j))
                yield (i, j)

    # Create generator and get N pairs
    pg = pair_generator()
    result_pairs = [next(pg) for _ in range(N)]
    return result_pairs


def annotate_pair_for_interestingsness_minihack(buffer_encoded_captions, pair_ids, mode="binary", buffer_inventory=None):
    annotations = []
    # Currently pair indices would come from here!
    # pair_indices = generate_pairs(num_rollouts, ep_length, int(num_pairs * 2))
    # each row contains: [rollout_id1, rollout_id2, t_1, t_2]

    for p_i in pair_ids:
        desc_encoded1 = buffer_encoded_captions[p_i[0], p_i[2], ...]
        desc_encoded2 = buffer_encoded_captions[p_i[1], p_i[3], ...]

        if mode == "binary":
            annotation_i = check_interestingness_binary(desc_encoded1, desc_encoded2)
        elif mode == "num_items":
            annotation_i = check_interestingness_num_items(desc_encoded1, desc_encoded2)
        elif mode == "distance":
            annotation_i = check_interestingness_location(desc_encoded1, desc_encoded2)
        elif mode == "inventory":
            inv_glyphs1 = buffer_inventory[p_i[0], p_i[2], ...]
            inv_glpyhs2 = buffer_inventory[p_i[1], p_i[3], ...]
            annotation_i = check_interestingness_inventory(desc_encoded1, desc_encoded2, inv_glyphs1, inv_glpyhs2)
        else:
            raise NotImplementedError

        annotations.append(annotation_i)
    
    return np.asarray(annotations)


