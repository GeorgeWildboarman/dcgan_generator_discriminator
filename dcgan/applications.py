import tensorflow as tf
from dcgan.dcgan4conv import model4conv

def conv4gen_celebA_trained():
  generator = model4conv().build_generator(activation='ReLU', dense=False)
  gen_checkpoint = tf.train.Checkpoint(generator)
  gen_save_path = 'dcgan/checkpoints/conv4gen_celebA_training_checkpoints-1'
  gen_checkpoint.restore(gen_save_path)
  return generator

def conv4disc_celebA_trained():
  disc_save_dir = 'dcgan/checkpoints/'
  print(os.path.exists(disc_save_dir))
  # discriminator = model4conv().build_discriminator(batchnorm=True, dropout=False, dense=False)
  # disc_checkpoint = tf.train.Checkpoint(discriminator)
  # disc_save_path = 'dcgan/checkpoints/conv4disc_celebA_training_checkpoints-1'
  # disc_save_path = '/dcgan/checkpoints/conv4disc_celebA_training_checkpoints-1'
  # disc_checkpoint.restore(disc_save_path)
  # return discriminator

