import tensorflow as tf
import numpy as np


class Objective(object):
    """Represents a generic objective function of an image (batch). Provides callability and basic
    arithmetic operations (+, -, * scalar) to instances.

    Attributes
    ----------
    func : function taking a 4D tensor to a 1D tensor
        Element-wise scalar function on a batch of images. The input is
        expected to have dimensions (batch, height, width, channel), and the
        output is expected to have only the batch dimension.
    batch : int or None
        Required batch size of images, if any.
    """
    
    def __init__(self, func, batch=None):
        self.func = func
        self.batch = batch
        
    def __call__(self, image_batch):
        if self.batch and tf.shape(image_batch)[0].numpy() != self.batch:
            raise Exception('batch size of images != required value of Objective')
        return self.func(image_batch)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise Exception('Objective only allowed to be multiplied by a scalar')
        def func(image_batch):
            return self.func(image_batch) * float(scalar)
        return Objective(func)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __add__(self, other):
        if self.batch and other.batch and self.batch != other.batch:
            raise Exception('Objectives of unmatched required batch sizes')
        def func(image_batch):
            return self.func(image_batch) + other.func(image_batch)
        batch = self.batch or other.batch
        return Objective(func, batch=batch)
    
    def __sub__(self, other):
        return self + other * -1


def L1(base=0.5):
    """Defines an objective function by the L1 norm of pixels with respect to a base value."""
    def func(image_batch):
        return tf.reduce_mean(tf.abs(image_batch - base), axis=[1, 2, 3])
    return Objective(func)


def TotalVar():
    """Defines an objective function by the total variation of pixels."""
    def func(image_batch):
        vol = tf.cast(tf.reduce_prod(tf.shape(image_batch)[1:]), tf.float32)
        return tf.image.total_variation(image_batch) / vol
    return Objective(func)


class LayerObjective(Objective):
    """Represents an objective function of an image (batch) determined by its activations in a
    specific layer of a specific model. Provides callability and basic arithmetic operations (+, -,
    * scalar) to instances, such that combinations of instances require only a single evaluation of
    the layer.

    Attributes
    ----------
    model : tf.keras.Model
        Keras model that takes images as inputs.
    layer : str
        Name of a layer in 'model'.
    layer_func : function taking a nD tensor (n>1) to a 1D tensor
        Element-wise scalar function on a batch of activations of 'layer'. The input is expected to
        have dimensions (batch, *output shape of 'layer'), and the output is expected to have only
        the batch dimension.
    batch : int or None
        Required batch size of images, if any.
    """
    
    def __init__(self, model, layer, layer_func, batch=None):
        self.model = model
        self.layer = layer
        i, o = model.input, model.get_layer(layer).output
        eval_layer = tf.keras.Model(inputs=i, outputs=o)
        self.layer_func = layer_func
        def func(image_batch):
            layer_output = eval_layer(image_batch)
            value = self.layer_func(layer_output)
            return value
        super().__init__(func, batch=batch)
        
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise Exception('LayerObjective only allowed to be multiplied by a scalar')
        def layer_func(layer_output):
            return self.layer_func(layer_output) * float(scalar)
        return LayerObjective(self.model, self.layer, layer_func, batch=self.batch)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __add__(self, other):
        if isinstance(other, LayerObjective):
            if self.model != other.model or self.layer != other.layer:
                raise Exception('LayerObjectives of unmatched models or layers')
            if self.batch and other.batch and self.batch != other.batch:
                raise Exception('LayerObjectives of unmatched required batch sizes')
            def layer_func(layer_output):
                return self.layer_func(layer_output) + other.layer_func(layer_output)
            batch = self.batch or other.batch
            return LayerObjective(self.model, self.layer, layer_func, batch=batch)
        elif isinstance(other, Objective):
            return super().__add__(other)
        else:
            raise Exception('operation not allowed')
        
    def __sub__(self, other):
        return self + other * -1
        

def LinearlyCombinedChannels(model, layer, channels, weights):
    """Defines an objective function by linear combinations of mean channel
    activations in a layer of a model.

    Parameters
    ----------
    model, layer : same as LayerObjective
    channels : list of int
        List of channel numbers.
    weights : 1D or 2D array of float
        Weights on 'channels', of dimensions (channel,) or (batch, channel).

    Returns
    -------
    LayerObjective
        Linear combination(s) of mean channel outputs as specified by 'channels' and 'weights'.
    """
    
    weights = np.array(weights)
    if len(weights.shape) not in (1, 2):
        raise Exception('rank of weights != 1 or 2' )
    if weights.shape[-1] != len(channels):
        raise Exception('innermost dimension of weights != number of channels')
    batch = None if len(weights.shape) == 1 else weights.shape[0]

    def layer_func(layer_output):
        channel_outputs = [tf.reduce_mean(layer_output[..., c], [1, 2]) for c in channels]
        channel_outputs = tf.stack(channel_outputs, axis=1)
        wt_channel_outputs = tf.reduce_sum(channel_outputs * weights, axis=-1)
        return wt_channel_outputs
    
    return LayerObjective(model, layer, layer_func, batch=batch)
    
   
def Channel(model, layer, channel):
    """Defines an objective function by the mean activation of a channel in a layer of a model."""
    return LinearlyCombinedChannels(model, layer, [channel], [1])


def Channels(model, layer, channels):
    """Defines an objective function by the mean activations of several channels in a layer of a
    model."""
    return LinearlyCombinedChannels(model, layer, channels, np.eye(len(channels)))


def InterpolatedChannels(model, layer, channel1, channel2, intervals=2):
    """Defines an objective function by linearly interpolating the mean activations of two channels
    in a layer of a model."""
    weights = [[1 - i / intervals, i / intervals] for i in range(intervals + 1)]
    return LinearlyCombinedChannels(model, layer, [channel1, channel2], weights)


def GramCosSim(model, layer):
    """Defines an objective function by the mean cosine similarity in Gram matrix with the other
    images in the batch."""
    def layer_func(layer_output):
        # batch / spatial / channel indices: a, b / i, j / p, q
        batch = tf.shape(layer_output).numpy()[0]
        gram = tf.einsum('aijp,aijq->apq', layer_output, layer_output)
        gram_norm = tf.sqrt(tf.einsum('apq,apq->a', gram, gram))
        gram_normalized = gram / (gram_norm[:, None, None] + 1e-10)
        gram_cossim = tf.einsum('apq,bpq->ab', gram_normalized, gram_normalized)
        # subtract cosine similarity with self (i.e. 1)
        gram_cossim -= tf.eye(batch)
        return tf.reduce_mean(gram_cossim, axis=1)
    return LayerObjective(model, layer, layer_func)


class LayerObjectiveFactory(object):
    """Represents a generator of objective functions based on the same layer of a model. An example
    of its usage is as follows:

        obj_gen = LayerObjectiveFactory(model, layer)
        obj = obj_gen(objectives.Channel, channel)
        obj -= obj_gen(objectives.GramCosSim) * 100
        ...

    In general, it can be called on (method, *args), where 'method' is any function taking (model,
    layer, *args) to a LayerObjective instance.
    """
    
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def __call__(self, method, *args):
        return method(self.model, self.layer, *args)
