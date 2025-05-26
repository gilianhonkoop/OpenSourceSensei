import itertools
import random
import numpy as np

# def check_interestingness_robodesk(end_effector, qpos_objects, qvel_objects):
#     # end_effector: self.physics.named.data.site_xpos['end_effector']
#     # qpos_objects: self.physics.data.qpos[self.num_joints:].copy()
#     # qvel_objects: self.physics.data.qvel[self.num_joints:].copy()

#     # qvel_objects_inds = {
#     #   "drawer_joint": 0,
#     #   "slide_joint": 1, 
#     #   "red_button": 2,
#     #   "green_button": 3,
#     #   "blue_button": 4, 
#     #   "ball": [8,9,10], #xyz positions
#     #   "upright_block": [14, 15, 16],
#     #   "flat_block": [20, 21, 22],
#     # }

#     qpos_objects_inds = {
#       "drawer_joint": 0,
#       "slide_joint": 1, 
#       "red_button": 2,
#       "green_button": 3,
#       "blue_button": 4, 
#       "ball": [8,9,10], #xyz positions
#       "upright_block": [15, 16, 17],
#       "flat_block": [22, 23, 24],
#     }

#     offset_drawer =  np.array([-1.59314870e-05,  5.29119446e-01,  6.64747558e-01])
#     offset_slide = np.array([-0.3,    0.78,   0.935])

    

#     return 0

def get_interaction_from_vel(qvel_objects):
    interaction_dict = {}
    qvel_objects_inds = {
      "drawer_joint": 0,
      "slide_joint": 1, 
      "red_button": 2,
      "green_button": 3,
      "blue_button": 4, 
      "ball": [8,9,10], #xyz positions
      "upright_block": [14, 15, 16],
      "flat_block": [20, 21, 22],
    }

    interactions = []
    for key, ind in qvel_objects_inds.items():
        if "ball" in key or "block" in key:
            interaction_dict[key] = np.any(np.abs(qvel_objects[ind])>2*1e-2, axis=-1)*1
        else:
            interaction_dict[key] = (np.abs(qvel_objects[ind])>2*1e-2) * 1
        interactions.append(interaction_dict[key])
    return np.asarray(interactions), interaction_dict    

def gripper_pos_to_target_distance(gripper_pos, target_pos):
    # Block_pos: nB (xnE) x horizon x 3*(nObj+1)
    return np.linalg.norm(gripper_pos - target_pos, axis=-1)

def stack_reward_from_obs(xpos_upright_block, xpos_flat_block):
    target_offset = [0, 0, 0.0377804]
    # current_offset = (self.physics.named.data.xpos['upright_block'] -
    #                   self.physics.named.data.xpos['flat_block'])

    current_offset = (xpos_upright_block - xpos_flat_block)
    offset_difference = np.linalg.norm(target_offset - current_offset)
    return (offset_difference < 0.04)*1

def is_state_interesting(obs_end_effector, obs_qpos_objects, obs_qvel_objects, pair_i, num):

    _, moved_objects_dict = get_interaction_from_vel(obs_qvel_objects)

    dist_thres = 15 * 1e-2 # 14.5 * 1e-2
    # -------- Check for task rewards -------------------
    qpos_objects_inds = {
        "drawer_joint": 0,
        "slide_joint": 1, 
        "red_button": 2,
        "green_button": 3,
        "blue_button": 4, 
        "ball": [8,9,10], #xyz positions
        "upright_block": [15, 16, 17],
        "flat_block": [22, 23, 24],
    }
    open_drawer = 1 * (obs_qpos_objects[qpos_objects_inds["drawer_joint"]] < -0.15)
    open_slide = 1 * (obs_qpos_objects[qpos_objects_inds["slide_joint"]]  > 0.45)

    stack = stack_reward_from_obs(obs_qpos_objects[qpos_objects_inds["upright_block"]],
                                       obs_qpos_objects[qpos_objects_inds["flat_block"]])
    

    # just for the ball double check that end effector is near the object!
    # although currently while objects fall it also counts as "interaction"
    ball_inds = qpos_objects_inds["ball"]
    dist2ball = gripper_pos_to_target_distance(
        gripper_pos=obs_end_effector[:3],
        target_pos=obs_qpos_objects[ball_inds],
    )
    close2ball = dist2ball < dist_thres
    # The and part is a stricter filtering than just overwriting it because
    # you get a positive signal for closeness even if you are very close to the object
    # but haven't moved it yet or you are holding it in your gripper but not moving much

    # moved_objects_indices[env.env_body_names.index("ball"), :] = np.logical_and(
    #     moved_objects_indices[env.env_body_names.index("ball"), :], close2ball
    # )
    moved_objects_dict["ball"] = close2ball and moved_objects_dict["ball"]

    # ==================================== ====================================
    # For the blocks also be less strict with an "or" statement!
    # ==================================== ====================================

    for block in ["upright_block", "flat_block"]:
        block_inds = qpos_objects_inds[block]
        dist2block = gripper_pos_to_target_distance(
            gripper_pos=obs_end_effector[:3],
            target_pos=obs_qpos_objects[block_inds],
        )

        block_dist_vec = np.abs(obs_end_effector[:3]-obs_qpos_objects[block_inds])
        close2block = (dist2block < dist_thres) or (dist2block < 0.21 and block_dist_vec[-1]<0.10 and np.sum(block_dist_vec[:2]<0.10)>=1)
        
        # close2block = dist2block < dist_thres

        # block_ind = list(moved_objects_dict.keys()).index(block)
        # moved_objects_indices[block_ind] = np.logical_or(
        #     moved_objects_indices[block_ind], close2block
        # )
        moved_objects_dict[block] = close2block or moved_objects_dict[block] 

    # -------------------------------------------------------------------------
    # ----------------------CHECK IF CLOSE TO DRAWER HANDLE--------------------
    # -------------------------------------------------------------------------

    offset_drawer =  np.array([-1.59314870e-05,  5.29119446e-01,  6.64747558e-01])

    drawer_pos = offset_drawer.copy()
    drawer_pos[1] += obs_qpos_objects[qpos_objects_inds["drawer_joint"]]
    
    dist2dhandle = gripper_pos_to_target_distance(
        gripper_pos=obs_end_effector[:3],
        target_pos=drawer_pos,
    )
    drawer_dist_vec = np.abs(obs_end_effector[:3]-drawer_pos)
    close2dhandle = (dist2dhandle < dist_thres) or (dist2dhandle < 0.22 and drawer_dist_vec[-1]<0.15 and np.sum(drawer_dist_vec[:2]<0.10)>=1)
    moved_objects_dict["drawer_joint"] =  moved_objects_dict["drawer_joint"] or close2dhandle

    # test = moved_objects_dict["drawer_joint"]
    # print(f"Pair {pair_i}, {num}: Distance to drawer handle: {dist2dhandle} , vec {drawer_dist_vec} and close2dhandle flag: {close2dhandle} and {test}")
    
    # -------------------------------------------------------------------------
    # ----------------------CHECK IF CLOSE TO SLIDE HANDLE---------------------
    # -------------------------------------------------------------------------

    offset_slide = np.array([-0.3,    0.78,   0.935])
    slide_pos = offset_slide.copy()
    slide_pos[0] += obs_qpos_objects[qpos_objects_inds["slide_joint"]]

    dist2shandle = gripper_pos_to_target_distance(
        gripper_pos=obs_end_effector[:3],
        target_pos=slide_pos,
    )
    slide_dist_vec = np.abs(obs_end_effector[:3]-slide_pos)
    close2shandle = (dist2shandle < dist_thres) or (dist2shandle < 0.21 and slide_dist_vec[-1]<0.05 and np.sum(slide_dist_vec[:2]<0.05)>=1)

    moved_objects_dict["slide_joint"] =  moved_objects_dict["slide_joint"] or close2shandle
    
    # test = moved_objects_dict["slide_joint"]
    # print(f"Pair {pair_i}, {num}: Distance to slide handle: {dist2shandle} , vec {slide_dist_vec} and and close2shandle flag: {close2shandle} and {test}")

    moved_objects_indices = np.asarray(list(moved_objects_dict.values()))
    # obj_interesting = np.concatenate([moved_objects_indices, open_drawer[None], stack[None], open_slide[None]], axis=0)
    obj_interesting = np.concatenate([moved_objects_indices, open_drawer[None], stack[None]], axis=0)
    return np.any(obj_interesting)


def annotate_pair_for_interestingsness_robodesk(buffer_end_effector, buffer_qpos_objects, buffer_qvel_objects, pair_ids):
    annotations = []
    # Currently pair indices would come from here!
    # pair_indices = generate_pairs(num_rollouts, ep_length, int(num_pairs * 2))
    # each row contains: [rollout_id1, rollout_id2, t_1, t_2]
    global_i = 0
    for p_i in pair_ids:
        ef1 = buffer_end_effector[p_i[0], p_i[2], ...]
        ef2 = buffer_end_effector[p_i[1], p_i[3], ...]

        qpos_objects1 = buffer_qpos_objects[p_i[0], p_i[2], ...]
        qpos_objects2 = buffer_qpos_objects[p_i[1], p_i[3], ...]

        qvel_objects1 = buffer_qvel_objects[p_i[0], p_i[2], ...]
        qvel_objects2 = buffer_qvel_objects[p_i[1], p_i[3], ...]

        interesting1 = is_state_interesting(ef1, qpos_objects1, qvel_objects1, global_i, 0)
        interesting2 = is_state_interesting(ef2, qpos_objects2, qvel_objects2, global_i, 1)

        if interesting1 and not interesting2:
            annotation_i = 0
        elif not interesting1 and interesting2:
            annotation_i = 1
        elif interesting1 and interesting2:
            annotation_i = 2
        else:
            annotation_i = 3    
        annotations.append(annotation_i)

        # print(f"for pair_i: {global_i}, int1: {interesting1}, int2: {interesting2}, annotation: {annotation_i}")
        global_i += 1

    return np.asarray(annotations)