<div align="center">

# Pantheon Lab Programming Assignment

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>


## ANSWERS 

1.  A generator is a neural network that creats fake samples, and the role of the discriminator is to identify these fake samples. During training, the generator tries to fool the discriminator into believing that these images are real and they belong to the dataset. In the given case, the generator will create fake MNIST samples (which comprises of the different digits) and the role of the discriminator is to identify these fake samples and avoid getting tricked. 


2. Noise in this case is the generator's input. The role of the generator here is to take this random vector (noise) and create a meaningful output. The meaningful output here is defined by the label which specifies which digit we are trying to generate through the given noise. For example, we feed the generator random noise with  the label 5: the generator will output a fake image which resembles the digit 5. The lable should be a real value or a true value, as we aim for the generator to give a high output, i.e. 1, (which is the probability of being a real image). 


3. This was my first time working on NN and high level python packages, and it was pretty fun. I was not sure how to proceed witht the test, as all the data and theory initially felt quiet overwhelming. However, according to my understanding of the methods I used to accomplish the task, below are the steps that should be taken care of:

a. Setting up an environment. It was initially difficult for me as I was using online python notebooks and google collab to run the python files. However after referring to sources online, I understood that the best way to start up with the work is by cloning thr github respository in the VSCode editor, and work on a development environment which is relevenat to the task, such as, conda. I ran the requirements.txt file to check the things I was missign. (Although it did not compile successfully, but through the process I could figure out the things I was missing).

b. After compiling the code I realised that I was missing a lot of libraries and packages that were required to be installed in the IDE, like omegaconf for the YAML file, hydra, dotenv, wandb etc. 

c. The final steps include Making the model (i.e filling out the missing code in the mnist_gan_model.py) and testing the model (this process failed for me due to IDE and configuration issues on my M1 macbook).


4. 

## What is all this?
This "programming assignment" is really just a way to get you used to
some of the tools we use every day at Pantheon to help with our research.

There are 4 fundamental areas that this small task will have you cover:

1. Getting familiar with training models using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

2. Using the [Hydra](https://hydra.cc/) framework

3. Logging and reporting your experiments on [weights and biases](https://wandb.ai/site)

4. Showing some basic machine learning knowledge

## What's the task?
The actual machine learning task you'll be doing is fairly simple! 
You will be using a very simple GAN to generate fake
[MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) images.

We don't excpect you to have access to any GPU's. As mentioned earlier this is just a task
to get you familiar with the tools listed above, but don't hesitate to improve the model
as much as you can!

## What you need to do

To understand how this framework works have a look at `src/train.py`. 
Hydra first tries to initialise various pytorch lightning components: 
the trainer, model, datamodule, callbacks and the logger.

To make the model train you will need to do a few things:

- [ ] Complete the model yaml config (`model/mnist_gan_model.yaml`)
- [ ] Complete the implementation of the model's `step` method
- [ ] Implement logging functionality to view loss curves 
and predicted samples during training, using the pytorch lightning
callback method `on_epoch_end` (use [wandb](https://wandb.ai/site)!) 
- [ ] Answer some questions about the code (see the bottom of this README)

**All implementation tasks in the code are marked with** `TODO`

Don't feel limited to these tasks above! Feel free to improve on various parts of the model

For example, training the model for around 20 epochs will give you results like this:

![example_train](./images/example_train.png)

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name pantheon-py38 python=3.8
conda activate pantheon-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default
python run.py experiment=train_mnist_gan.yaml

# train on CPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=0

# train on GPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=1
```

You can override any parameter from command line like this
```yaml
python run.py experiment=train_mnist_gan.yaml trainer.max_epochs=20 datamodule.batch_size=32
```

The current state of the code will fail at
`src/models/mnist_gan_model.py, line 29, in configure_optimizers`
This is because the generator and discriminator are currently assigned `null`
in `model/mnist_gan_model.yaml`. This is your first task in the "What you need to do" 
section.

## Bonus tasks

- **Implement your own networks**: you are free to choose what you deem most appropriate, but we recommend using CNN and their variants
- Use a more complex dataset, such as Fashion-MNIST

## Questions

Try to prepare some short answers to the following questions below for discussion in the interview.

* What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

* The generator network in this code base takes two arguments: `noise` and `labels`.
What are these inputs and how could they be used at inference time to generate an image of the number 5?

* What steps are needed to deploy a model into production?

* If you wanted to train with multiple GPUs, 
what can you do in pytorch lightning to make sure data is allocated to the correct GPU? 

## Submission

- Using git, keep the existing git history and add your code contribution on top of it. Follow git best practices as you see fit. We appreciate readability in the commits
- Add a section at the top of this README, containing your answers to the questions, as well as the output `wandb` graphs and images resulting from your training run. You are also invited to talk about difficulties you encountered and how you overcame them
- Link to your git repository in your email reply and share it with us/make it public

<br>
