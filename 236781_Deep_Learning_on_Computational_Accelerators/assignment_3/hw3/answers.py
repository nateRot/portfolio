r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 128
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.01
    hypers['lr_sched_factor'] = 0.002
    hypers['lr_sched_patience'] = 0.0001
    # raise NotImplementedError()
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.4
    start_seq = 'ACT I. SCENE 1. Rousillon. The COUNT\'S palace'
    # raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
Had we not split the corpus into sequences, characters from very early on would have weight on characters at the end of the corpus.
Though this is illogical, the context of characters from far earlier states have very little (if at all) impact on what we expect 
the next char to be. Therefore we split into sequences to limit the contributing factors in our prediction only to the most significant 
characters.
"""

part1_q2 = r"""
Multiple hidden states are stored between batches, when a new batch is created we calculate along with the sotred hidden states.
"""

part1_q3 = r"""
Since hidden states are being passed between batches, shuffling them would disorganize and reset the hidden state, the outcome would be a decrease 
in performance.
"""

part1_q4 = r"""
1. Lower temperature will grant lower variance between the generated word and the sequence. When training we wish to battle over fitting, this 
can be done inducing randomness into the mode, by setting the temperature.

2. Higher temperatures create more noise. Nonsensical words, undefined chars, etc. Higher temperatures grant higher variance allowing the model 
more diverse characters (words) to select. May lead to undesired results and poor performance.

3. We get the counter effect for lower temperatures, the variance decreases which in turn generates more common chars (words), lower the diversity, and 
increasing repetition.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 512
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 0.001
    hypers['learn_rate'] = 1e-4
    hypers['betas'] = (0.5, 0.999)
    # raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
$\sigma^2$ hyperparameter allows us to twek the tradeoff between the reconstruction loss and the KLD loss. A large $\sigma^2$ causes the data loss element 
to decrease. The generated images will have greater resemblance to the dataset images. The generated images will be of better quality (closer to dataset 
images), yet the variaty will be more limited.
Smaller $\sigma^2$ values on the other hand grant an increase to data loss element, in turn allows more "randomness" to the generated images. They will be 
as a result of lower quality but greater variety.

"""

part2_q2 = r"""
1. Reconstruction loss is the distance between the reconstructed image and the original. KL divergence loss measures between the encoder 
probability function and latent space prior.

2. The KL loss term is used to determine how close the Encoder's distribution to the Latent space prior. If the term is low we will see a 
guassian distribution $\mu = 0$  $\sigma = 1$ for the Encoder

3. This allows us to sample from the latent space while making few assumptions about the model. The encoder would calculate the mean and varience 
of its distribution.

"""

part2_q3 = r"""
Assuming the latent space to be of a normal distribution, images closer in latent space to real images, are more likely to themselves be similar 
to real images. By increasing the evidence distribution the generative model would produce images of better quality.
$p(\bb{X})$ is the probability of generating images X with our generator. Therefore if we maximize the probability over **real** images, we can 
expect images the resemble **real** images to increase. 

"""

part2_q4 = r"""
Defining $\bb{\sigma}^2_{\bb{\alpha}}$ to be positive-real number, this can be guaranteed by use of the ReLu activation function, though as we 
know around zero, the gradient is not defined. Floating point arithmetic has instability when dealing with small numbers, in most cases the 
standard deviation values are small. If we opt to use the log tranform, we can map the smaller numbers onto a larger interval. This would grant 
greater stability to Floating Point arithmetic. Yet something else, the $\log{\sigma^2}$ value in VAE's are used in the Kullback-Leibler divergence 
term.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 128
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.0002
    hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.0002
    hypers['generator_optimizer']['betas'] = (0.5, 0.999)
    # raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
When training we want to ensure the gradient observed with the coupled model (Generator & Discriminator) does not propagate back to the Genrator. 
So we decouple the pair and train the Dicriminator only, on the forward pass. Next we train the Generator while keeping the gradients for calculating 
the loss
"""

part3_q2 = r"""
1. As the Discriminator has itself not been trained the loss by the Generator may be low, while the images still are of poor quality. We overcame 
this by designating a minimum number of epochs before we begin to test for low loss. We tried to think of more "elegant" solutions to this but haven't 
found one.

2. If such a case happens it means that the Discriminator is getting better at identifying *real* images, while simutaneously identifying generated 
images as *real* as well

"""

part3_q3 = r"""
We can see that our GAN images are still missing sharpness, yet they are less blurry than the ones the VAE model outputed. We can explain this in part 
do to the greater variance which gives us in comparison, sharper edges.

"""

# ==============
