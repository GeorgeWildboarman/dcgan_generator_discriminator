# gan-generator-discriminator

DCGAN (Deep Convolutional Generative Adversarial Network).
  DCGAN is a type of generative model that uses deep convolutional neural networks
  for both the generator and discriminator components.
  It was introduced in the 2015 paper titled
  "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
  by Radford et al.
  
  DCGAN has been widely used for generating realistic images across various domains,
  such as generating human faces, landscapes, and even artistic images. 

  In a DCGAN (Deep Convolutional Generative Adversarial Network),
  there are two main components: the generator and the discriminator.

    Generator: The generator takes random noise (latent space vectors) as input and generates synthetic/generated images.
    It typically consists of layers such as dense (fully connected) layers, transposed convolutional layers, batch normalization layers,
    and activation functions like ReLU or Tanh. The goal of the generator is to learn to generate realistic images that resemble the training data.

    Discriminator: The discriminator takes an image (real or generated) as input and determines whether it is real or fake.
    It consists of convolutional layers, batch normalization layers, and activation functions like LeakyReLU.
    The discriminator is trained to distinguish between real images from the training dataset and fake/generated images produced by the generator.

  The build_generator method constructs the generator model using transposed convolutional layers.
  The build_discriminator method constructs the discriminator model using convolutional layers.
