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


def validate(reward_model, loss_fn, validation_loader, pref_type_key, device, iteration, val_met):
    cur_val_loss = 0.0
    validation_acc = 0.0
    score_validation_acc = 0.0

    for mb in tqdm.tqdm(validation_loader):
        labels = mb[pref_type_key].to(device).type(torch.float)
        # Label greater than 1 indicates no preference
        labels[torch.where(labels > 1)[0]] = 0.5

        with torch.no_grad():
            result = reward_model.forward_pairs(mb)
        # sequence length x BS x 2
        rewards = result.rewards
        rewards = rewards.mean(axis=0)

        soft_labels = torch.zeros(len(rewards), 2, device=device)
        soft_labels[:, 1] = labels
        soft_labels[:, 0] = 1.0 - labels
        predicted_log_probs = nn.functional.log_softmax(rewards, dim=1)

        val_loss = loss_fn(predicted_log_probs, soft_labels)
        cur_val_loss += val_loss.item()

        # Only measure accuracy on pairs where the annotator has a preference
        reward_argmax = np.argmax(rewards.detach().cpu().numpy(), axis=1)
        labels = labels.cpu().numpy()
        validation_acc += np.mean(reward_argmax[labels != 0.5] == labels[labels != 0.5])
        # score_labels = np.argmax(mb["score"], axis=1).cpu().numpy()
        # score_validation_acc += np.mean(reward_argmax == score_labels)

    # Save and log validation metrics
    val_met.iter.append(iteration)
    # val_met.score_validation_accs.append(score_validation_acc / len(validation_loader))
    val_met.validation_accs.append(validation_acc / len(validation_loader))
    val_met.total_val_loss.append(cur_val_loss / len(validation_loader))

    # log.info(
    #     f"Iteration {iteration} "
    #     f"Score Validation accuracy: {val_met.score_validation_accs[-1]:.3f}\n"
    #     f"Iteration {iteration} "
    #     f"Validation accuracy: {val_met.validation_accs[-1]:.3f}\n"
    #     f"Iteration {iteration} "
    #     f"Validation loss: {val_met.total_val_loss[-1]:.3f}"
    # )

    return val_met

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
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardModelTrainer:
    def __init__(
        self,
        train_params,
        reward_model,
        dataset_dir="/media/csancaktar/Elements/motif_dataset",
        seed=42,
    ) -> None:
        self.reward_model = reward_model
        self.device = self.reward_model.device
        self.dataset_dir = dataset_dir
        self._parse_train_params(**train_params)
        self.seed = seed
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=self.reward_lr, weight_decay=self.weight_decay
        )
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        # print("self.logger dir", self.logger.logdir)
        log_dir = os.path.join(self.logger.logdir, "events_writer")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    @torch.no_grad()
    def _update_normalizer(self, all_obs_dict):
        if self.reward_model.encoder.use_obs_vec:
            self.reward_model.encoder.obs_vec_normalizer.update(all_obs_dict["observations"])
        if self.reward_model.encoder.use_clip_embedding_right:
            self.reward_model.encoder.clip_right_normalizer.update(all_obs_dict["clip_embedding_cam_right"])
        if self.reward_model.encoder.use_clip_embedding_left:
            self.reward_model.encoder.clip_left_normalizer.update(all_obs_dict["clip_embedding_cam_left"])

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
        data_keys = ["observations"]
        if self.reward_model.encoder.use_clip_embedding_right:
            data_keys.append("clip_embedding_cam_right")
        if self.reward_model.encoder.use_clip_embedding_left:
            data_keys.append("clip_embedding_cam_left")
        if self.reward_model.encoder.use_image:
            data_keys.append("images")

        preference_keys = ["preferences_gt"]

        print("Loading dataset...")
        dataset = PairsDataset(
            self.dataset_dir,  # "/media/csancaktar/Elements/motif_dataset",
            data_keys=data_keys,
            info_keys=info_keys,
            preference_keys=preference_keys,
            transform=transforms.Compose([ToTensorDict()]),
            mode="r",
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(
            dataset=dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )

        print("Loaded dataset!")

        # --------- update normalizer ---------
        if self.reward_model.encoder.normalize_inputs:
            self._update_normalizer(
                {k: v[train_dataset.indices, ...].reshape(-1, v.shape[-1]) for k, v in dataset.data_arrays.items()}
            )

        pref_type_key = f"{preference_keys[0]}_pref"
        collate_ignore_keys = ["idx", pref_type_key]
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=partial(flatten_pair_collate_fn, ignore_keys=collate_ignore_keys),
            num_workers=10,
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=partial(flatten_pair_collate_fn, ignore_keys=collate_ignore_keys),
            num_workers=10,
        )

        self.reward_model.model_to_device(self.device)

        loss_fn = lambda logprobs, target: -(target * logprobs).sum() / logprobs.shape[0]

        # Define metrics
        train_metrics = {
            "epoch": [],
            "total_train_loss": [],
            "total_train_acc": [],
        }
        train_met = DictAsAttributes(train_metrics)
        val_metrics = {
            "iter": [],
            "total_val_loss": [],
            # "score_validation_accs": [],
            "validation_accs": [],
        }
        val_met = DictAsAttributes(val_metrics)

        # Training
        num_iter = 0
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            reward_rms = RunningMeanStd(self.device)

            for i, mb in enumerate(tqdm.tqdm(train_loader)):
                if num_iter % len(train_loader) == 0:
                    # full_reward_rms, all_msgs_rewards = get_rms(cfg, reward_model, train_loader, all_msgs_input, device)

                    # log.info(
                    #     f"\n Full Reward mean: {full_reward_rms.mean[0]:.3f} "
                    #     f"Full Reward variance: {full_reward_rms.var[0]:.3f}"
                    # )

                    val_met = validate(
                        self.reward_model,
                        loss_fn,
                        validation_loader,
                        pref_type_key,
                        self.device,
                        num_iter,
                        val_met,
                    )
                    # self.logger.log(val_met.total_val_loss[-1], key="validation/epoch_loss")
                    # self.logger.log(val_met.validation_accs[-1], key="validation/validation_accs")
                    self.writer.add_scalar("validation/epoch_loss", val_met.total_val_loss[-1], num_iter)
                    self.writer.add_scalar("validation/validation_accs", val_met.validation_accs[-1], num_iter)
                    # save(
                    #     cfg,
                    #     num_iter,
                    #     reward_model,
                    #     optimizer,
                    #     train_met._data_dict,
                    #     val_met._data_dict,
                    #     full_reward_rms,
                    #     all_msgs_rewards,
                    # )

                result = self.reward_model.forward_pairs(mb)
                rewards = result.rewards  # sequence length x BS x 2
                reward_rms.update(rewards.reshape(-1, 1).detach())
                rewards = rewards.mean(axis=0)

                labels = mb[pref_type_key].to(self.device).type(torch.float)  # BS
                labels[torch.where(labels > 1)[0]] = 0.5
                soft_labels = torch.zeros(len(rewards), 2, device=self.device)
                soft_labels[:, 1] = labels
                soft_labels[:, 0] = 1.0 - labels

                predicted_log_probs = nn.functional.log_softmax(rewards, dim=1)
                loss = loss_fn(predicted_log_probs, soft_labels)

                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()  # Perform back-propagation
                self.optimizer.step()  # Update the weights

                reward_argmax = np.argmax(rewards.detach().cpu().numpy(), axis=1)
                labels = labels.cpu().numpy()
                train_acc += np.mean(reward_argmax[labels != 0.5] == labels[labels != 0.5])
                cur_acc = train_acc / (i + 1)

                num_iter += 1

            train_met.total_train_loss.append(train_loss / len(train_loader))
            train_met.total_train_acc.append(train_acc / len(train_loader))
            train_met.epoch.append(epoch)

            # self.logger.log(train_loss / len(train_loader), key="train/epoch_loss")
            # self.logger.log(train_acc / len(train_loader), key="train/train_accs")
            self.writer.add_scalar("train/epoch_loss", train_loss / len(train_loader), num_iter)
            self.writer.add_scalar("train/train_accs", train_acc / len(train_loader), num_iter)

            print(f"Training loss at epoch {epoch} is: {train_loss / len(train_loader)}")
            print(f"Validation loss at epoch {epoch} is: {val_met.total_val_loss[-1]}")

            if epoch % 1 == 0 and save_cpt:
                self.save(os.path.join(working_dir, f"checkpoint_{epoch}"))

        self.writer.close()

    # Save model parameters
    def save(self, path):
        torch.save(
            {
                "model": self.reward_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                **(
                    {"obs_vec_normalizer": self.reward_model.encoder.obs_vec_normalizer.state_dict()}
                    if self.reward_model.encoder.normalize_inputs and self.reward_model.encoder.use_obs_vec
                    else {}
                ),
                **(
                    {"clip_right_normalizer": self.reward_model.encoder.clip_right_normalizer.state_dict()}
                    if self.reward_model.encoder.normalize_inputs and self.reward_model.encoder.use_clip_embedding_right
                    else {}
                ),
                **(
                    {"clip_left_normalizer": self.reward_model.encoder.clip_left_normalizer.state_dict()}
                    if self.reward_model.encoder.normalize_inputs and self.reward_model.encoder.use_clip_embedding_left
                    else {}
                ),
            },
            path,
        )

    def load(self, path):
        state_dicts = torch.load(path)
        self.reward_model.load(path)
        self.optimizer.load_state_dict(state_dicts["optimizer"])
