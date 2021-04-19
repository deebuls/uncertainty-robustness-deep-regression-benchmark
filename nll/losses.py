import tensorflow as tf
import numpy as np

def laplace_nll_loss(input, target, scale, eps=1e-06, reduction='mean'):
    
    # Inputs and target should be of same shape
    input = tf.reshape(input, [tf.shape(input).numpy()[0], -1])
    target = tf.reshape(target, [tf.shape(target).numpy()[0], -1])
    if input.shape != target.shape:
        raise ValueError("input and target must have same size")

    # Second dim of scale must match the input size
    scale = tf.reshape(scale, [tf.shape(scale).numpy()[0], -1])
    if scale.shape != target.shape:
        raise ValueError("scale must have same size as input")

    #check validity of reduction mode
    if reduction !='none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + "is not valid" )

    #Entries of scale must be non negative
    tf.debugging.assert_non_negative(
        scale, message="scale has negative numbers", summarize="have you missed to make scale positive", name=None  
    )

    # Clamp for stability
    scale = tf.identity(scale)
    scale = tf.stop_gradient(scale)
    scale = tf.clip_by_value(scale, clip_value_min=eps, clip_value_max=100)

    #Calculate loss
    loss = (tf.math.log(2*scale) + tf.abs(input - target)/scale)
    loss = tf.reshape(loss, [tf.shape(input).numpy()[0], -1])
    loss = tf.reduce_sum(loss, axis=1)

    #Apply reduction
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)
    else:
        return loss

