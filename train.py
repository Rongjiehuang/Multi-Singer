#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Multi-Singer."""

import argparse
import logging
import ipdb
import os
import sys

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml
import losses

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import optimizers

from datasets import AudioMelEmbedDataset
from datasets import Embeds_Collater
from layers import PQMF
from losses import MultiResolutionSTFTLoss
from utils import read_hdf5
import os
from utils import simple_table
from encoder import inference as encoder


# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Multi-Singer training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x = []

        x.append(batch['feats'])
        embed = batch['embed'].to(self.device)

        y = batch['audios'].to(self.device)
        x = tuple([x_.to(self.device) for x_ in x])

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x).to(self.device)

        # Singer Perceptual loss
        embed_loss = 0
        spk_similariy = []
        for i in range(y_.shape[0]):
            preprocess_ = encoder.preprocess_wav_torch(y_[i]).squeeze()
            loss_embed_ = encoder.embed_utterance_torch(preprocess_)
            preprocess = encoder.preprocess_wav_torch(y[i]).squeeze()
            loss_embed = encoder.embed_utterance_torch(preprocess)
            embed_loss += self.criterion["mse"](loss_embed,loss_embed_)

        self.total_train_loss["train/embed_loss"] += embed_loss.item()
        self.total_train_loss["train/spk_similariy"] = np.mean(np.array(spk_similariy))
        gen_loss = self.config["lambda_embed"] * embed_loss

        # multi-resolution sfft loss

        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        gen_loss += sc_loss + mag_loss


        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_)
            embed_p_ = self.model["embed_discriminator"](y_, embed)

            uncondition_adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            self.total_train_loss["train/uncondition_adv_loss"] += uncondition_adv_loss.item()

            # Embed discriminator loss
            speaker_condition_adv_loss = self.criterion["mse"](embed_p_, embed_p_.new_ones(embed_p_.size()))
            self.total_train_loss["train/speaker_condition_adv_loss"] += speaker_condition_adv_loss.item()
            adv_loss = uncondition_adv_loss + speaker_condition_adv_loss

            # add adversarial loss to generator loss
            gen_loss += self.config["lambda_adv"] * adv_loss

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"] and self.steps % self.config["interval"] == 0:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)

            # discriminator loss
            embed_p = self.model["embed_discriminator"](y, embed)
            embed_p_ = self.model["embed_discriminator"](y_.detach(), embed)

            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())

            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            uncondition_discriminator_loss = real_loss + fake_loss

            embed_real_loss = self.criterion["mse"](embed_p, embed_p.new_ones(embed_p.size()))
            embed_fake_loss = self.criterion["mse"](embed_p_, embed_p_.new_zeros(embed_p_.size()))
            speaker_condition_discriminator_loss = embed_real_loss + embed_fake_loss

            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/uncondition_discriminator_loss"] += uncondition_discriminator_loss.item()
            self.total_train_loss["train/speaker_condition_discriminator_loss"] += speaker_condition_discriminator_loss.item()
            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            uncondition_discriminator_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

            # update discriminator
            self.optimizer["embed_discriminator"].zero_grad()
            speaker_condition_discriminator_loss.backward()
            if self.config["embed_discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["embed_discriminator"].parameters(),
                    self.config["embed_discriminator_grad_norm"])
            self.optimizer["embed_discriminator"].step()
            self.scheduler["embed_discriminator"].step()
        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x = []


        if self.config['use_noise_input']:
            x.append(batch['noise'])
        if self.config['use_f0']:
            x.append(batch['f0_origins'])
        if self.config['use_chroma']:
            x.append(batch['chromas'])
        x.append(batch['feats'])

        y = batch['audios'].to(self.device)
        x = tuple([x_.to(self.device) for x_ in x])
        embed = batch['embed'].to(self.device)
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x).to(self.device)

        # Singer Perceptual loss
        embed_loss = 0
        spk_similariy = []
        for i in range(y_.shape[0]):
            preprocess_ = encoder.preprocess_wav_torch(y_[i]).squeeze()
            loss_embed_ = encoder.embed_utterance_torch(preprocess_)
            preprocess = encoder.preprocess_wav_torch(y[i]).squeeze()
            loss_embed = encoder.embed_utterance_torch(preprocess)
            embed_loss += self.criterion["mse"](loss_embed,loss_embed_)
            spk_similariy.append(cosine_similarity(loss_embed_.reshape(1,256).cpu().numpy(), loss_embed.reshape(1,256).cpu().numpy()))

        gen_loss = self.config["lambda_embed"] * embed_loss
        self.total_eval_loss["eval/embed_loss"] += embed_loss.item()
        self.total_eval_loss["eval/spk_similariy"] = np.mean(np.array(spk_similariy))

        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        gen_loss += sc_loss + mag_loss

        # Unconditional/Conditional Loss
        p_ = self.model["discriminator"](y_)
        embed_p_ = self.model["embed_discriminator"](y_, embed)

        uncondition_adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
        gen_loss += self.config["lambda_adv"] * uncondition_adv_loss
        speaker_condition_adv_loss = self.criterion["mse"](embed_p_, embed_p_.new_ones(embed_p_.size()))
        gen_loss += self.config["lambda_adv"] * speaker_condition_adv_loss

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y)
        p_ = self.model["discriminator"](y_)

        embed_p = self.model["embed_discriminator"](y, embed)
        embed_p_ = self.model["embed_discriminator"](y_, embed)

        # Unconditional Loss
        real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
        fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
        uncondition_discriminator_loss = real_loss + fake_loss

        # Conditional Loss
        embed_real_loss = self.criterion["mse"](embed_p, embed_p.new_ones(embed_p.size()))
        embed_fake_loss = self.criterion["mse"](embed_p_, embed_p_.new_zeros(embed_p_.size()))
        speaker_condition_discriminator_loss = embed_real_loss + embed_fake_loss

        # add to total eval loss
        self.total_eval_loss["eval/uncondition_adv_loss"] += uncondition_adv_loss.item()
        self.total_eval_loss["eval/speaker_condition_adv_loss"] += speaker_condition_adv_loss.item()
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/uncondition_discriminator_loss"] += uncondition_discriminator_loss.item()
        self.total_eval_loss["eval/speaker_condition_discriminator_loss"] += speaker_condition_discriminator_loss.item()

    def _eval_epoch(self):

        """Evaluate model one epoch."""

        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x = []
        if self.config['use_noise_input']:
            x.append(batch['noise'])

        x.append(batch['feats'])
        y_batch = batch['audios'].to(self.device)
        x_batch = tuple([x_.to(self.device) for x_ in x])

        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname,exist_ok=True)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

    def _adjust_length(self, x, c):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Singer.")
    parser.add_argument("--inputdir",'-i', type=str, required=True,
                        help="directory to input feats .")
    parser.add_argument("--outdir",'-o',type=str, default="checkpoints/",
                        help="directory to save checkpoints.")
    parser.add_argument("--config",'-c',type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume",'-r', default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank","--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()



    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("USE CPU")
    else:
        print("USE GPU %d"%args.rank)
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    # if args.rank != 0:
    #     sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = 0.0  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        frames_threshold = config["batch_max_steps"] // config["hop_size"] + \
            2 * config["generator_params"].get("aux_context_window", 0)
    else:
        frames_threshold = None

    train_text = os.path.join(args.inputdir, 'train.txt')

    train_dataset = AudioMelEmbedDataset(
        root_file=train_text,
        feat_type=config.get("feat_type", "librosa"),
        frames_threshold=frames_threshold,
        use_f0=config['use_f0'],
        use_chroma=config['use_chroma'],
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )

    logging.info(f"The number of training files = {len(train_dataset)}.")

    dev_text = os.path.join(args.inputdir, 'dev.txt')
    dev_dataset = AudioMelEmbedDataset(
        root_file=dev_text,
        feat_type=config.get("feat_type", "librosa"),
        frames_threshold=frames_threshold*10,
        use_f0=config['use_f0'],
        use_chroma=config['use_chroma'],
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )

    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Embeds_Collater(
        batch_max_steps=config["batch_max_steps"],
        out_dim=config["generator_params"]["out_channels"],
        hop_size=config["hop_size"],
        # keep compatibility
        aux_context_window=config["generator_params"].get("aux_context_window", 0),
        # keep compatibility
        use_noise_input=config['use_noise_input'],
        use_f0=config['use_f0'],
        use_chroma=config['use_chroma']
    )
    eval_collater = Embeds_Collater(
        batch_max_steps=10*config["batch_max_steps"],
        out_dim=config["generator_params"]["out_channels"],
        hop_size=config["hop_size"],
        # keep compatibility
        aux_context_window=config["generator_params"].get("aux_context_window", 0),
        # keep compatibility
        use_noise_input=config['use_noise_input'],
        use_f0=config['use_f0'],
        use_chroma=config['use_chroma']
    )
    sampler = {"train": None, "dev": None}

    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=eval_collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }


    # define models and optimizers
    generator_class = getattr(
        models,
        # keep compatibility
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    discriminator_class = getattr(
        models,
        # keep compatibility
        config.get("discriminator_type", "SC_ParallelWaveGANDiscriminator_01"),
    )
    embed_discriminator_class = getattr(
        models,
        # keep compatibility
        config.get("embed_discriminator_type", "SC_ParallelWaveGANDiscriminator_01"),
    )
    loss_class = getattr(
        losses,
        # keep compatibility
        config.get("loss_type", "MultiResolutionSTFTLoss"),
    )
    model = {
        "generator": generator_class(
            **config["generator_params"]).to(device),
        "discriminator": discriminator_class(
            **config["discriminator_params"]).to(device),
        "embed_discriminator": embed_discriminator_class(
            **config["embed_discriminator_params"]).to(device),
    }
    criterion = {
        "stft": loss_class(**config["stft_loss_params"]).to(device),
        "mse": torch.nn.MSELoss().to(device),
    }

    if config.get("use_feat_match_loss", False):  # keep compatibility
        criterion["l1"] = torch.nn.L1Loss().to(device)
    if config["generator_params"]["out_channels"] > 1:
        criterion["pqmf"] = PQMF(
            subbands=config["generator_params"]["out_channels"],
            # keep compatibility
            **config.get("pqmf_params", {})
        ).to(device)
    if config.get("use_subband_stft_loss", False):  # keep compatibility
        assert config["generator_params"]["out_channels"] > 1
        criterion["sub_stft"] = MultiResolutionSTFTLoss(
            **config["subband_stft_loss_params"]).to(device)

    generator_optimizer_class = getattr(
        optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    discriminator_optimizer_class = getattr(
        optimizers,
        # keep compatibility
        config.get("discriminator_optimizer_type", "RAdam"),
    )
    embed_discriminator_optimizer_class = getattr(
        optimizers,
        # keep compatibility
        config.get("embed_discriminator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
        "embed_discriminator": embed_discriminator_optimizer_class(
            model["embed_discriminator"].parameters(),
            **config["embed_discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("discriminator_scheduler_type", "StepLR"),
    )
    embed_discriminator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("embed_discriminator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
        "embed_discriminator": embed_discriminator_scheduler_class(
            optimizer=optimizer["embed_discriminator"],
            **config["embed_discriminator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])
        model["embed_discriminator"] = DistributedDataParallel(model["embed_discriminator"])
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    logging.info(model["embed_discriminator"])

    simple_table([
        ('PreprocessData Path', args.inputdir),
        ('Checkpoints Path', args.outdir),
        ('Config File', args.config),
        ('Generator File', config["generator_type"]),
        ('Discriminator File', config["discriminator_type"]),
    ])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # load encoder
    encoder.load_model(config["enc_model_fpath"],rank=args.rank)
    logging.info(f"Successfully load parameters from %s."%config["enc_model_fpath"])

    for param in encoder._model.parameters():
        param.requires_grad = False

    encoder.num_params()

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
