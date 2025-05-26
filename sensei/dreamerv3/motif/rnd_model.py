import torch
from torch import nn
import copy

from motif.model.model import TorchRewardEncoder
from motif.utils.dict_attribute import AttrDict

# # From: https://github.com/rll-research/url_benchmark/blob/main/agent/ddpg.py
# class Encoder(nn.Module):
#     def __init__(self, obs_shape):
#         super().__init__()

#         assert len(obs_shape) == 3
#         self.repr_dim = 32 * 35 * 35

#         self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#                                      nn.ReLU())

#         self.apply(self.initialize_weights)

#     def forward(self, obs):
#         obs = obs / 255.0 - 0.5
#         h = self.convnet(obs)
#         h = h.view(h.shape[0], -1)
#         return h

#     def initialize_weights(self, layer):
#         # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
#         gain = self.param_init_gain

#         if self.param_initialization == "orthogonal":
#             if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
#                 nn.init.orthogonal_(layer.weight.data, gain=gain)
#                 layer.bias.data.fill_(0)
#             else:
#                 # LSTMs and GRUs initialize themselves
#                 # should we use orthogonal/xavier for LSTM cells as well?
#                 # I never noticed much difference between different initialization schemes, and here it seems safer to
#                 # go with default initialization,
#                 pass
#         elif self.param_initialization == "xavier_uniform":
#             if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
#                 nn.init.xavier_uniform_(layer.weight.data, gain=gain)
#                 layer.bias.data.fill_(0)
#             else:
#                 pass

class RND(nn.Module):
    def __init__(self, model_params, device=torch.device("cuda:0")):
        super().__init__()
        self._parse_model_params(**model_params)
        self.device = device

        self.encoder = TorchRewardEncoder(model_params.encoder_params, self.device)

        self.encoders = [self.encoder] # completely useless actually...
        self.core_output_size = self.encoder.encoder_out_size
        self.reward_fn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.core_output_size, 1),
        )

        self.predictor = nn.Sequential(self.encoder, 
                                       nn.ReLU(),
                                       nn.Linear(self.core_output_size, self.hidden_dim),
                                       nn.ReLU(),
                                    #    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    #    nn.ReLU(),
                                       nn.Linear(self.hidden_dim, self.rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(self.encoder),
                                    nn.ReLU(),
                                    nn.Linear(self.core_output_size, self.hidden_dim),
                                    nn.ReLU(),
                                    # nn.Linear(self.hidden_dim, self.hidden_dim),
                                    # nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(self.initialize_weights)
        self.train()
        self.model_to_device(self.device)

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)
        self.predictor.to(device)
        self.target.to(device)

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
        hidden_dim=1024,
        rnd_rep_dim=512,
        normalize_inputs=False,
        param_initialization="orthogonal",
        param_init_gain=1.0,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.rnd_rep_dim = rnd_rep_dim
        self.normalize_inputs = normalize_inputs
        self.param_initialization = param_initialization
        self.param_init_gain = param_init_gain

    # def forward_head(self, obs_dict):
    #     x = self.encoder(obs_dict)
    #     return x

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

        # x = self.forward_head(mb)

        prediction, target = self.predictor(mb), self.target(mb)

        prediction_error = torch.square(target.detach() - prediction).mean(
                    dim=-1, keepdim=True)
        result = AttrDict(
            dict(
                rewards=prediction_error,
            )
        )
        return result

    # def forward(self, obs):
    #     obs = self.aug(obs)
    #     obs = self.normalize_obs(obs)
    #     obs = torch.clamp(obs, -self.clip_val, self.clip_val)
    #     prediction, target = self.predictor(obs), self.target(obs)
    #     prediction_error = torch.square(target.detach() - prediction).mean(
    #         dim=-1, keepdim=True)
    #     return prediction_error


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

if __name__ == "__main__":
    pass