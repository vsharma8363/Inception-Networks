# Building the Vehicle Recognition Project

1. The Inception Module in Tensorflow
2. Building/Sourcing the dataset
3. Constructing the model
4. Training and improving the accuracy
5. Final Testing
6. Exporting the model
7. Running via Android Studio (OpenCV + Tensorflow libraries)

## 1. The Inception Module

* Below is an introduction to GoogLeNet and the inception module

### GoogLeNet
* Developed by Google, architecture won first place in 2014
* 21 times lighter than VGG-16 above
#### The Inception Module
* The key breakthrough of the GoogLeNet architecture was the development of the **inception module**
    * A sub-network that can perform parallel processing
    * The input is put through two layers and the output of each is concatenated together in a process called **depth concatenation**
##### Versions of the Inception Module
![Inception Modules](https://miro.medium.com/max/2698/1*aq4tcBl9t5Z36kTDeZSOHA.png)
* 1x1 Convolutions are called **bottlenecks** that perform compression in order to reduce parametric weight
##### Average Pooling
* Introducing average pooling after the convolutional block further reduces parametric weight and the computational advantage is incredible --> You do lose some info, but not enough to make the addition of AvgPooling not worth it

### Contructing the Network with Tensorflow and Keras
* Taking an object-oriented approach to the implementation of the Inception Module using the Functional API of Keras

## 2. Building/Sourcing the Dataset
* Following the traditional process of constructing a Tensorflow Dataset using images sourced from the SmartCNNs project on Github



```python
directory = 'animals' # Directory of the animal images
animals = ['cat', 'butterfly', 'dog', 'sheep', 'spider', 'chicken', 'horse', 'squirrel', 'cow', 'elephant']
num_classes = len(animals)

# Tensorflow version I plan on using for this project
import tensorflow as tf
import os
print(tf.__version__)
```

    2.1.0



```python
list_ds = tf.data.Dataset.list_files(str(directory + '/*/*')) # /*/* go down to the files
for f in list_ds.take(5):
  print(f.numpy())

IMG_HEIGHT = 224
IMG_WIDTH = 224
```

    b'animals/chicken/OIP-mFN57MsrJHRtpCcbq51anQHaCh.jpeg'
    b'animals/sheep/OIP-i-ni93yhowK-kkjLzFNJCQHaEK.jpeg'
    b'animals/squirrel/OIP-1L7YAkx0PfjhEef7yXhtPgHaFh.jpeg'
    b'animals/spider/OIP-Y_-Ao5Q1BENkq32g--lUcQHaE9.jpeg'
    b'animals/squirrel/OIP-1SVdvhAO-68BYuw0hj-3kAAAAA.jpeg'



```python
def decode_img(img):
  # convert the raw string into a 3d tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def get_label_image_pair(file_path):
    
    # Find the class name -----------------------------
    segments = tf.strings.split(file_path, os.path.sep)
    # The second to last is the directory name
    tensor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask = segments[-2] == animals
    label = tf.boolean_mask(tensor, mask) # CONVERT TO ONE-HOT
    
    # Get the image in raw format ---------------------
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

labeled_ds = list_ds.map(get_label_image_pair) #num_parallel_calls=tf.data.experimental.AUTOTUNE)

labeled_ds = labeled_ds.shuffle(buffer_size=1000).batch(32)
    
for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", len(label.numpy()), label.dtype)
```

    Image shape:  (32, 224, 224, 3)
    Label:  32 <dtype: 'int32'>



```python
import matplotlib.pyplot as plt

def central_crop_transform(image, label):
  image = tf.image.central_crop(image, central_fraction=0.6)
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  return image, label

def rotations_transform(image, label):
  image = tf.image.flip_up_down(image)
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  return image, label

def brightness_transform(image, label):
  image = tf.image.adjust_brightness(image, 0.25)
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  return image, label
    
all_datasets = [labeled_ds, labeled_ds.map(central_crop_transform), 
                labeled_ds.map(rotations_transform), labeled_ds.map(brightness_transform)]

```

## 2.5 Implementing the Inception Module using Keras Functional API
* The Naive Inception Module, other versions have the same applied theory with slight alterations


```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate

def naive_inception_module(prev_layer, filters = [64, 128, 32]):
    conv_1by1 = Conv2D(filters[0], kernel_size = (1,1), padding = 'same', activation = 'relu')(prev_layer)
    conv_3by3 = Conv2D(filters[1], kernel_size = (3,3), padding = 'same', activation = 'relu')(prev_layer)
    conv_5by5 = Conv2D(filters[2], kernel_size = (5,5), padding = 'same', activation = 'relu')(prev_layer)
    conv_pooled = MaxPooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    return concatenate([conv_1by1, conv_3by3, conv_5by5, conv_pooled], axis = -1) # -1 for depth concat

```

## 3. Constructing the model

I am going to use the inception modules to create a CNN for this project, utilizing both bottlenecks and dropout to improve model accuracy and reduce computational complexity


```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

# Built using Functional so I can use the naive inception module defined above
model_inputs = Input(shape = [input_shape])
conv1 = Conv2D(32, kernel_size=(5,5), input_shape=input_shape)(model_inputs)
conv2 = Conv2D(32, kernel_size=(3,3), input_shape=input_shape)(conv1)
conv3 = Conv2D(32, kernel_size=(1,1), input_shape=input_shape)(conv2)
pooling = MaxPooling2D(pool_size=(2,2))(conv3)
naive = naive_inception_module(pooling)
conv4 = Conv2D(32, kernel_size=(3,3), input_shape=input_shape)(naive)
conv5 = Conv2D(32, kernel_size=(1,1), input_shape=input_shape)(conv4)
pooling2 = MaxPooling2D(pool_size=(2,2))(conv5)
predictions = Dense(num_classes, activation='softmax')(Flatten()(pooling2))
model = Model(inputs = model_inputs, outputs = predictions)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])
```


```python

```


```python
model.fit(all_datasets[0], epochs=5, verbose=1)
```

    Train for 819 steps
    Epoch 1/5
     24/819 [..............................] - ETA: 1:17:38 - loss: 4.7542 - accuracy: 0.1780


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-13-0f58e2de3ebd> in <module>
    ----> 1 model.fit(all_datasets[0], epochs=5, verbose=1)
    

    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        817         max_queue_size=max_queue_size,
        818         workers=workers,
    --> 819         use_multiprocessing=use_multiprocessing)
        820 
        821   def evaluate(self,


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        340                 mode=ModeKeys.TRAIN,
        341                 training_context=training_context,
    --> 342                 total_epochs=epochs)
        343             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
        344 


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        126         step=step, mode=mode, size=current_batch_size) as batch_logs:
        127       try:
    --> 128         batch_outs = execution_function(iterator)
        129       except (StopIteration, errors.OutOfRangeError):
        130         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in execution_function(input_fn)
         96     # `numpy` translates Tensors to values in Eager mode.
         97     return nest.map_structure(_non_none_constant_value,
    ---> 98                               distributed_function(input_fn))
         99 
        100   return execution_function


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in __call__(self, *args, **kwds)
        566         xla_context.Exit()
        567     else:
    --> 568       result = self._call(*args, **kwds)
        569 
        570     if tracing_count == self._get_tracing_count():


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in _call(self, *args, **kwds)
        597       # In this case we have created variables on the first call, so we run the
        598       # defunned version which is guaranteed to never create variables.
    --> 599       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        600     elif self._stateful_fn is not None:
        601       # Release the lock early so that multiple threads can perform the call


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in __call__(self, *args, **kwargs)
       2361     with self._lock:
       2362       graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 2363     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       2364 
       2365   @property


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _filtered_call(self, args, kwargs)
       1609          if isinstance(t, (ops.Tensor,
       1610                            resource_variable_ops.BaseResourceVariable))),
    -> 1611         self.captured_inputs)
       1612 
       1613   def _call_flat(self, args, captured_inputs, cancellation_manager=None):


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1690       # No tape is watching; skip to running the function.
       1691       return self._build_call_outputs(self._inference_function.call(
    -> 1692           ctx, args, cancellation_manager=cancellation_manager))
       1693     forward_backward = self._select_forward_and_backward_functions(
       1694         args,


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        543               inputs=args,
        544               attrs=("executor_type", executor_type, "config_proto", config),
    --> 545               ctx=ctx)
        546         else:
        547           outputs = execute.execute_with_cancellation(


    ~/opt/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         59     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,
         60                                                op_name, inputs, attrs,
    ---> 61                                                num_outputs)
         62   except core._NotOkStatusException as e:
         63     if name is not None:


    KeyboardInterrupt: 



```python

```
