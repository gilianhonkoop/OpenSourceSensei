import torch
from torch import nn

from motif.model.model_utils import ConvEncoder, Normalizer, build_mlp


class TorchRewardEncoder(nn.Module):
    def __init__(self, model_params, device):
        super().__init__()

        self._parse_model_params(**model_params)
        # self.timing = timing

        # Define network
        out_dim = 0

        if self.use_obs_vec:
            # self.obs_vec_size = 176  ## put in the environment
            if self.normalize_inputs:
                self.obs_vec_normalizer = Normalizer(self.obs_vec_size, eps=1e-6, device=device)

            self.encoder_obs = build_mlp(
                input_dim=self.obs_vec_size,
                output_dim=self.encoder_hidden_dim,
                size=self.encoder_hidden_dim,
                num_layers=self.encoder_num_layers,
                activation="relu",
                layer_norm=False,
            )

            out_dim += self.encoder_hidden_dim

        if self.use_clip_embedding_right:
            self.clip_vec_size = 1280  ## put in the environment

            if self.normalize_inputs:
                self.clip_right_normalizer = Normalizer(self.clip_vec_size, eps=1e-6, device=device)

            self.encoder_clip_right = build_mlp(
                input_dim=self.clip_vec_size,
                output_dim=self.encoder_hidden_dim,
                size=self.encoder_hidden_dim,
                num_layers=self.encoder_num_layers,
                activation="relu",
                layer_norm=False,
            )

            out_dim += self.encoder_hidden_dim

        if self.use_clip_embedding_left:
            self.clip_vec_size = 1280  ## put in the environment

            if self.normalize_inputs:
                self.clip_left_normalizer = Normalizer(self.clip_vec_size, eps=1e-6, device=device)

            self.encoder_clip_left = build_mlp(
                input_dim=self.clip_vec_size,
                output_dim=self.encoder_hidden_dim,
                size=self.encoder_hidden_dim,
                num_layers=self.encoder_num_layers,
                activation="relu",
                layer_norm=False,
            )

            out_dim += self.encoder_hidden_dim

        if self.use_image:
            input_ch = 3
            if self.start_stride==2:
                conv_filters = [
                    [input_ch, self.image_filter_max // 4, 8, 2],
                    [self.image_filter_max // 4, self.image_filter_max // 2, 4, 1],
                    [self.image_filter_max // 2, self.image_filter_max, 3, 1],
                ]
            else:
                # og settings with stride 4
                conv_filters = [
                    [input_ch, self.image_filter_max // 4, 8, 4],
                    [self.image_filter_max // 4, self.image_filter_max // 2, 4, 2],
                    [self.image_filter_max // 2, self.image_filter_max, 3, 2],
                ]
            # conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
            if self.resize_image:
                input_shape = (3, 64, 64)
            else:
                input_shape = (3, self.image_resolution, self.image_resolution)
                # input_shape=(3, 224, 224)

            self.encoder_img = ConvEncoder(
                activation="elu",
                conv_filters=conv_filters,
                fc_layer_size=0,
                encoder_extra_fc_layers=0,
                input_shape=input_shape,
                downsample=self.cnn_downsample,
            )

            out_dim += (
                self.encoder_img.conv_head_out_size
                if self.encoder_img.fc_layer_size == 0
                else self.encoder_img.fc_layer_size
            )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.encoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_dim // 2, self.encoder_hidden_dim),
        )

        self.encoder_out_size = self.encoder_hidden_dim

    def _parse_model_params(
        self,
        *,
        encoder_num_layers,
        encoder_hidden_dim,
        # param_initialization="orthogonal",
        # param_init_gain=1.0,
        use_obs_vec=False,
        use_image=False,
        use_clip_embedding_right=False,
        use_clip_embedding_left=False,
        normalize_inputs=True,
        resize_image=False,
        image_filter_max=128,
        image_resolution=224,
        obs_vec_size=176,
        start_stride=4,
        cnn_downsample=True,
        **kwargs,
    ):
        # Necessary to get the observation dimension
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        # self.param_initialization = param_initialization
        # self.param_init_gain = param_init_gain

        self.use_obs_vec = use_obs_vec
        self.use_image = use_image
        self.use_clip_embedding_right = use_clip_embedding_right
        self.use_clip_embedding_left = use_clip_embedding_left
        self.normalize_inputs = normalize_inputs
        self.resize_image = resize_image
        self.image_filter_max = image_filter_max
        self.image_resolution = image_resolution
        self.obs_vec_size = obs_vec_size
        self.start_stride = start_stride
        self.cnn_downsample = cnn_downsample

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        """Default implementation, can be overridden in derived classes."""
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def get_encoder_out_size(self):
        return self.encoder_out_size

    def forward(self, obs_dict):
        # clip_embedding_cam_right = obs_dict["clip_embedding_cam_right"]
        # batch_size = clip_embedding_cam_right.shape[0]

        reps = []

        if self.use_obs_vec:
            observations = obs_dict["observations"].float()
            if self.normalize_inputs:
                observations = self.obs_vec_normalizer.normalize(observations)
            observations_rep = self.encoder_obs(observations)
            reps.append(observations_rep)

        if self.use_clip_embedding_right:
            clip_embedding_cam_right = obs_dict["clip_embedding_cam_right"].float()
            if self.normalize_inputs:
                clip_embedding_cam_right = self.clip_right_normalizer.normalize(clip_embedding_cam_right)
            clip_right_rep = self.encoder_clip_right(clip_embedding_cam_right)
            reps.append(clip_right_rep)

        if self.use_clip_embedding_left:
            clip_embedding_cam_left = obs_dict["clip_embedding_cam_left"].float()
            if self.normalize_inputs:
                clip_embedding_cam_left = self.clip_left_normalizer.normalize(clip_embedding_cam_left)
            clip_left_rep = self.encoder_clip_left(clip_embedding_cam_left)
            reps.append(clip_left_rep)

        if self.use_image:
            images = obs_dict["images"].float() / 255
            if self.resize_image:
                images = nn.functional.interpolate(images, 64, mode="bilinear", align_corners=False)
            image_rep = self.encoder_img(images)
            reps.append(image_rep)
        # -- [batch size x K]
        reps = torch.cat(reps, dim=1)
        st = self.fc(reps)

        return st
