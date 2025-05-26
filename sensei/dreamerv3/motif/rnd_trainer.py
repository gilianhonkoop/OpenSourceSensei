import os
from functools import partial

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms

from motif import allogger
from motif.dataset import PairsDataset, flatten_pair_collate_fn
from motif.utils.preprocessing import DictAsAttributes, ToTensorDict


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, device, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RNDTrainer:
    def __init__(
        self,
        train_params,
        rnd_model,
        dataset_dir="/media/csancaktar/Elements/motif_dataset",
        preference_key="preferences_llava",
        seed=42,
    ) -> None:
        self.rnd_model = rnd_model
        self.device = self.rnd_model.device
        self.dataset_dir = dataset_dir
        self.preference_key = preference_key
        self._parse_train_params(**train_params)
        self.seed = seed
        self.optimizer = torch.optim.Adam(
            self.rnd_model.parameters(),
            lr=self.reward_lr,
            weight_decay=self.weight_decay,
        )
        self.logger = allogger.get_logger(
            scope=self.__class__.__name__, default_outputs=["tensorboard"]
        )
        # print("self.logger dir", self.logger.logdir)
        log_dir = os.path.join(self.logger.logdir, "events_writer")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    @torch.no_grad()
    def _update_normalizer(self, all_obs_dict):
        if self.rnd_model.encoder.use_obs_vec:
            self.rnd_model.encoder.obs_vec_normalizer.update(
                all_obs_dict["observations"]
            )
        if self.rnd_model.encoder.use_clip_embedding_right:
            self.rnd_model.encoder.clip_right_normalizer.update(
                all_obs_dict["clip_embedding_cam_right"]
            )
        if self.rnd_model.encoder.use_clip_embedding_left:
            self.rnd_model.encoder.clip_left_normalizer.update(
                all_obs_dict["clip_embedding_cam_left"]
            )

    def _parse_train_params(
        self,
        *,
        learning_rate,
        batch_size,
        num_epochs,
        weight_decay,
        **kwargs,
    ):
        self.reward_lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay

    def train(self, save_cpt=False, working_dir=None):
        info_keys = None
        data_keys = []
        # data_keys = ["observations"]
        if self.rnd_model.encoder.use_obs_vec:
            data_keys.append("observations")
        if self.rnd_model.encoder.use_clip_embedding_right:
            data_keys.append("clip_embedding_cam_right")
        if self.rnd_model.encoder.use_clip_embedding_left:
            data_keys.append("clip_embedding_cam_left")
        if self.rnd_model.encoder.use_image:
            data_keys.append("images")

        # preference_keys = ["preferences_gt"]
        preference_keys = [self.preference_key]

        print("Loading dataset...")
        dataset = PairsDataset(
            self.dataset_dir,
            data_keys=data_keys,
            info_keys=info_keys,
            preference_keys=preference_keys,
            transform=transforms.Compose([ToTensorDict()]),
            mode="r",
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        print("Loaded dataset!")

        # --------- update normalizer ---------
        if self.rnd_model.encoder.normalize_inputs:
            self._update_normalizer(
                {
                    k: v[train_dataset.indices, ...].reshape(-1, v.shape[-1])
                    for k, v in dataset.data_arrays.items()
                }
            )

        pref_type_key = f"{preference_keys[0]}_pref"
        collate_ignore_keys = ["idx", pref_type_key]
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=partial(
                flatten_pair_collate_fn, ignore_keys=collate_ignore_keys
            ),
            num_workers=10,
        )

        # validation_loader = torch.utils.data.DataLoader(
        #     validation_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     collate_fn=partial(
        #         flatten_pair_collate_fn, ignore_keys=collate_ignore_keys
        #     ),
        #     num_workers=10,
        # )

        self.rnd_model.model_to_device(self.device)

        # Define metrics
        train_metrics = {
            "epoch": [],
            "total_train_loss": [],
        }
        train_met = DictAsAttributes(train_metrics)
        # val_metrics = {
        #     "iter": [],
        #     "total_val_loss": [],
        # }
        # val_met = DictAsAttributes(val_metrics)

        # Training
        num_iter = 0
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            reward_rms = RunningMeanStd(self.device)

            for i, mb in enumerate(tqdm.tqdm(train_loader)):

                result = self.rnd_model.forward(mb)
                rewards = result.rewards  # BS x 2
                reward_rms.update(rewards.reshape(-1, 1).detach())

                loss = rewards.mean()

                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()  # Perform back-propagation
                self.optimizer.step()  # Update the weights

                num_iter += 1

            train_met.total_train_loss.append(train_loss / len(train_loader))
            train_met.epoch.append(epoch)

            # self.logger.log(train_loss / len(train_loader), key="train/epoch_loss")
            self.writer.add_scalar(
                "train/epoch_loss", train_loss / len(train_loader), num_iter
            )

            print(
                f"Training loss at epoch {epoch} is: {train_loss / len(train_loader)}"
            )

            if epoch % 1 == 0 and save_cpt:
                self.save(os.path.join(working_dir, f"checkpoint_{epoch}"))

        self.writer.close()

    # Save model parameters
    def save(self, path):
        torch.save(
            {
                "model": self.rnd_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                **(
                    {
                        "obs_vec_normalizer": self.rnd_model.encoder.obs_vec_normalizer.state_dict()
                    }
                    if self.rnd_model.encoder.normalize_inputs
                    and self.rnd_model.encoder.use_obs_vec
                    else {}
                ),
                **(
                    {
                        "clip_right_normalizer": self.rnd_model.encoder.clip_right_normalizer.state_dict()
                    }
                    if self.rnd_model.encoder.normalize_inputs
                    and self.rnd_model.encoder.use_clip_embedding_right
                    else {}
                ),
                **(
                    {
                        "clip_left_normalizer": self.rnd_model.encoder.clip_left_normalizer.state_dict()
                    }
                    if self.rnd_model.encoder.normalize_inputs
                    and self.rnd_model.encoder.use_clip_embedding_left
                    else {}
                ),
            },
            path,
        )

    def load(self, path):
        state_dicts = torch.load(path)
        self.rnd_model.load(path)
        self.optimizer.load_state_dict(state_dicts["optimizer"])
