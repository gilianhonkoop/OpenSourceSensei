import torch
from torch import nn

from motif.model.model import TorchRewardEncoder
from motif.utils.dict_attribute import AttrDict


class RewardModel(nn.Module):
    def __init__(self, model_params, device=torch.device("cuda:0")):
        super().__init__()
        self._parse_model_params(**model_params)
        self.device = device

        self.encoder = TorchRewardEncoder(model_params.encoder_params, self.device)
        self.encoders = [self.encoder]
        self.core_output_size = self.encoder.encoder_out_size
        self.reward_fn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.core_output_size, 1),
        )
        self.apply(self.initialize_weights)
        self.train()  # eval() for inference?
        self.model_to_device(self.device)

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoders[0].device_and_type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.param_init_gain

        if self.param_initialization == "orthogonal":
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            else:
                # LSTMs and GRUs initialize themselves
                # should we use orthogonal/xavier for LSTM cells as well?
                # I never noticed much difference between different initialization schemes, and here it seems safer to
                # go with default initialization,
                pass
        elif self.param_initialization == "xavier_uniform":
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            else:
                pass

    def _parse_model_params(
        self,
        *,
        param_initialization="orthogonal",
        param_init_gain=1.0,
        **kwargs,
    ):
        self.param_initialization = param_initialization
        self.param_init_gain = param_init_gain

    def forward_head(self, obs_dict):
        x = self.encoder(obs_dict)
        return x

    def forward(self, mb, add_dim=False):
        for key, value in mb.items():
            if key in [
                "observations",
                "clip_embedding_cam_right",
                "clip_embedding_cam_left",
                "images",
            ]:
                if add_dim:
                    mb[key] = torch.tensor(value[None, ...]).to(self.device)
                else:
                    mb[key] = value.to(self.device)

        x = self.forward_head(mb)
        rewards = self.reward_fn(x)

        result = AttrDict(
            dict(
                rewards=rewards,
            )
        )
        return result

    def forward_pairs(self, mb):
        data_key = None
        for key, value in mb.items():
            if key in [
                "observations",
                "clip_embedding_cam_right",
                "clip_embedding_cam_left",
                "images",
            ]:
                mb[key] = value.to("cuda")
                if data_key is None:
                    # This is to just get one of the existing data keys
                    data_key = key
        # batch_size = len(mb["observations"].reshape(-1, 2, mb["observations"].shape[-1]))
        batch_size = mb[data_key].shape[0] // 2
        x = self.forward_head(mb)
        x = x.reshape(batch_size * 2, -1)  # Batch size x 2, *
        rewards = self.reward_fn(x)
        rewards = rewards.reshape(1, batch_size, 2)  # batch size, 2

        result = AttrDict(
            dict(
                rewards=rewards,
            )
        )
        return result

    # Save model parameters
    def save(self, path):
        torch.save(
            {
                "model": self.state_dict(),
                **(
                    {"obs_vec_normalizer": self.encoder.obs_vec_normalizer.state_dict()}
                    if self.encoder.normalize_inputs and self.encoder.use_obs_vec
                    else {}
                ),
                **(
                    {
                        "clip_right_normalizer": self.encoder.clip_right_normalizer.state_dict()
                    }
                    if self.encoder.normalize_inputs
                    and self.encoder.use_clip_embedding_right
                    else {}
                ),
                **(
                    {
                        "clip_left_normalizer": self.encoder.clip_left_normalizer.state_dict()
                    }
                    if self.encoder.normalize_inputs
                    and self.encoder.use_clip_embedding_left
                    else {}
                ),
            },
            path,
        )

    def load(self, path):
        state_dicts = torch.load(path)
        self.load_state_dict(state_dicts["model"])

        if self.encoder.normalize_inputs:
            if self.encoder.use_obs_vec:
                self.encoder.obs_vec_normalizer.load_state_dict(
                    state_dicts["obs_vec_normalizer"]
                )
            if self.encoder.use_clip_embedding_right:
                self.encoder.clip_right_normalizer.load_state_dict(
                    state_dicts["clip_right_normalizer"]
                )
            if self.encoder.use_clip_embedding_left:
                self.encoder.clip_left_normalizer.load_state_dict(
                    state_dicts["clip_left_normalizer"]
                )
