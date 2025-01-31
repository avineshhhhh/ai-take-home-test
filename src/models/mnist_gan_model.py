from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        raise NotImplementedError

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        # TODO: implement the step method of the GAN model.
        #     : This function should return both a dictionary of losses
        #     : and current loss of the network being optimised.
        #     :
        #     : When training with pytorch lightning, because we defined 2 optimizers in
        #     : the `configure_optimizers` function above, we use the `optimizer_idx` parameter
        #     : to keep a track of which network is being optimised.
        
        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # TODO: Create adversarial ground truths
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # TODO: Create noise and labels for generator input
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        if optimizer_idx == 0 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the generator
            # HINT: when optimizer_idx == 0 the model is optimizing the generator
            raise NotImplementedError

            # TODO: Generate a batch of images
            self.generated_imgs = self(z)

            # TODO: Calculate loss to measure generator's ability to fool the discriminator
            loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": loss}
            log_dict = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            

        if optimizer_idx == 1 or not self.training:
            # TODO: generate images and calculate the adversarial loss for the discriminator
            # HINT: when optimizer_idx == 1 the model is optimizing the discriminator
            raise NotImplementedError

            # TODO: Generate a batch of images
            self.generated_imgs = self(z)

            # TODO: Calculate loss for real images
            real_images_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # TODO: Calculate loss for fake images
            fakes = torch.zeros(imgs.size(0), 1)
            fakes = fakes.type_as(imgs)

            fake_images_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fakes)

            # TODO: Calculate total discriminator loss
            sum_of_losses = (real_images_loss + fake_images_loss)
            discr_loss = sum_of_losses / 2
            tqdm_dict = {"d_loss": discr_loss}
            log_dict = OrderedDict({"loss": discr_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

        return log_dict, loss

    def on_epoch_end(self):
        # TODO: implement functionality to log predicted images to wandb
        #     : at the end of each epoch
        
        wandb.log(key="predicted images", images=self.generated_imgs)

        # TODO: Create fake images
        
        z = self.validation_z.type_as(self.generator.model[0].weight)
        fake_imgs = self(z)
        
        fakes = torch.zeros(fake_imgs.size(0), 1)
        fakes = fakes.type_as(fake_imgs)
        

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # TODO: log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)
                #     : replace `None` with your wandb Image object
                logger.experiment.log({"gen_imgs": fakes})
