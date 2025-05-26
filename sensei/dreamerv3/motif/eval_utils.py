import numpy as np

# import open_clip
import torch

# og cam from top but closer!
camera_1 = {
    "name": "top",
    "azimuth": 90.0,
    "distance": 1.55,
    "elevation": -45.0,
}

# from right side! (bin side)
camera_2 = {
    "name": "right",
    "azimuth": 180.0,
    "distance": 1.55,
    "elevation": -18,
}

# from left side (button side!)
camera_3 = {
    "name": "left",
    "azimuth": 25.0,
    "distance": 1.05,
    "elevation": -15,
}

cameras = [camera_1, camera_2, camera_3]


def set_camera(env, camera):
    frame = env.render("rgb_array", 224, 224)
    env.viewer.cam.azimuth = camera["azimuth"]
    env.viewer.cam.distance = camera["distance"]
    env.viewer.cam.elevation = camera["elevation"]
    env.sim.forward()
    return


def get_batch_dict_for_reward_model(reward_model, rollout_images=None, rollout_obs=None, rollout_images_left=None):
    mb = {}
    if reward_model.encoder.use_obs_vec:
        assert rollout_obs is not None
        mb["observations"] = torch.from_numpy(rollout_obs).float().to(reward_model.device)
    if reward_model.encoder.use_image:
        assert rollout_images is not None
        if isinstance(rollout_images, torch.Tensor):
            images = rollout_images.permute(0, 3, 1, 2).float().to(reward_model.device)
        else:
            images = np.asarray(rollout_images).copy().transpose(0, 3, 1, 2)
            images = torch.from_numpy(images).float().to(reward_model.device)
        mb["images"] = images
    if reward_model.encoder.use_clip_embedding_right or reward_model.encoder.use_clip_embedding_left:
        # removed CLIP for now to not deal with an extra dependency!
        raise NotImplementedError
        # clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-bigG-14", "laion2b_s39b_b160k")
        # # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        # clip_model.to(reward_model.device)

        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     if reward_model.encoder.use_clip_embedding_right:
        #         assert rollout_images is not None
        #         clip_embeddings_right = torch.zeros((len(rollout_images), 1280), device=reward_model.device)
        #         for i, img in enumerate(rollout_images):
        #             if isinstance(img, np.ndarray):
        #                 img = Image.fromarray(img)
        #             image = preprocess(img).unsqueeze(0)
        #             image_features = clip_model.encode_image(torch.tensor(image, device=reward_model.device))
        #             clip_embeddings_right[i, :] = image_features
        #         mb["clip_embedding_cam_right"] = clip_embeddings_right
        #     if reward_model.encoder.use_clip_embedding_left:
        #         assert rollout_images_left is not None
        #         clip_embeddings_left = torch.zeros((len(rollout_images), 1280), device=reward_model.device)
        #         for i, img in enumerate(rollout_images_left):
        #             if isinstance(img, np.ndarray):
        #                 img = Image.fromarray(img)
        #             image = preprocess(img).unsqueeze(0)
        #             image_features = clip_model.encode_image(torch.tensor(image, device=reward_model.device))
        #             clip_embeddings_left[i, :] = image_features
        #         mb["clip_embedding_cam_left"] = clip_embeddings_left
    return mb


def reward_model_with_mb(reward_model, mb):
    reward_model.model_to_device(reward_model.device)
    reward_model.eval()
    with torch.no_grad():
        return reward_model.forward(mb)
