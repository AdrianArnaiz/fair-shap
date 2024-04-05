import tensorflow as tf
import pandas as pd

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_data_loader(df,
                    label,
                    partition="train",
                    input_shape=(128,128), 
                    batch_size=128,
                    augment = False,
                    weight = None, 
                    crop = None):
  """_summary_

  Args:
      df (_type_): _description_
      label (_type_): _description_
      partition (str, optional): _description_. Defaults to "train".
      input_shape (tuple, optional): _description_. Defaults to (128,128).
      batch_size (int, optional): _description_. Defaults to 128.
      augment (bool, optional): _description_. Defaults to False.
      weight (_type_, optional): _description_. Defaults to None.
  """
  def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    #Crop
    if crop=='LFWA':
      img = tf.image.crop_to_bounding_box(img, 60, 60, 130, 130)
    elif crop=='CelebA':
      img = tf.image.crop_to_bounding_box(img, 70, 35, 108, 108)
    elif not crop is None:
      raise Exception(f'Crop value not known - crop:{crop}')

    # Resize the image to the desired size
    img = tf.image.resize(img, input_shape)
    img /=255.0
    return img
  
  def process_path(file_path, gender):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, gender

  def process_path_weighted(file_path, gender, weight):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, gender, weight

  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
      tf.keras.layers.experimental.preprocessing.RandomZoom(0.3, 0.2),],)

  #TDS Start
  if weight is None:
    ds = tf.data.Dataset.from_tensor_slices((df.path, df[label]))
    #Read images and normalize 
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
  else:
    #https://www.tensorflow.org/guide/keras/train_and_evaluate#sample_weights
    ds = tf.data.Dataset.from_tensor_slices((df.path, df[label], df[weight]))
    #Read images and normalize 
    ds = ds.map(process_path_weighted, num_parallel_calls=AUTOTUNE)

  if partition == "train":
    ds = ds.shuffle(buffer_size=1000)

  # Cache 
  ds = ds.cache()

  # Batch
  if partition == "train":
    ds = ds.repeat()
  ds = ds.batch(batch_size = batch_size)

  if partition == "train" and augment == True:
    if weight is None:
      ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
    else:
      ds = ds.map(lambda x, y, z: (data_augmentation(x), y, z), num_parallel_calls=AUTOTUNE)
  
  ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
  return ds
