import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mini_lucid_tf2 import params, transformations


def render_vis(objective, img_size, batch=None, optimizer=None, steps=(200,),
               freq_decay=1.0, rgb_corr=True, transforms=None, display_plots=False):
    """Generates and displays an image (batch) that maximizes an objective function.

    Parameters
    ----------
    objective : objectives.Objective
        Objective of visualization.
    img_size : int
        Size of each image (both its height and width).
    batch : int
        Batch size of images. If a specific value is required by 'objective', it is overridden
        accordingly. Otherwise, in the case of None, it defaults to 1.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer. In the case of None, Adam with a learning rate of 0.05 is used (as in Lucid).
    steps : tuple of ints
        Optimization steps at which intermediate results are displayed. The largest number also
        determines the number of optimization steps.
    freq_decay : float
        Frequency decay rate, controlling the downscaling of high frequency modes. (See
        params.ImageParam.)
    rgb_corr: bool
        Whether to impose empirical RGB correlations. (See params.ImageParam.)
    transforms : list of functions taking an image to an image
        Perturbations to be applied to the images at each optimization step. In the case of None, a
        default list is used.
    display_plots : bool
        Whether to display plots of convergence progress, i.e. mean absolute change of pixels and
        objective values.
    
    Returns
    -------
    img_batch : 4D tensor
        Visualization of the specified objective as given by the optimization result, of the shape
        (batch size, image size, image size, 3).
    """

    batch = objective.batch or batch or 1
    transforms = transforms or transformations.default_list()
    optimizer = optimizer or tf.keras.optimizers.Adam(0.05)
    
    # initialize an image parametrization
    img_param = params.ImageParam(img_size, batch=batch, init_noise=0.01, 
                                  freq_decay=freq_decay, rgb_corr=rgb_corr)
    
    # initialize arrays for mean pixel changes and objective values
    var_path = np.zeros((batch, max(steps) - 1))
    obj_path = np.zeros((batch, max(steps)))
    
    for i in range(max(steps)):
        
        with tf.GradientTape() as tape:
            
            # evaluate the images from parameters
            img_batch = img_param.eval()
            
            if i in steps:
                print(f'Step {i}')
                display_image_batch(img_batch)
            
            # record the mean pixel changes over this step
            if i > 0:
                var = tf.abs(img_batch - prev_img_batch)
                var_path[:, i-1] = tf.reduce_mean(var, [1, 2, 3]).numpy()
            prev_img_batch = img_batch
            
            # apply perturbations and restore the image size
            for transform in transforms:
                img_batch = transform(img_batch)
            img_batch = tf.image.resize(img_batch, [img_size, img_size])
    
            # compute and record objective value
            obj_val = objective(img_batch)
            obj_path[:, i] = obj_val.numpy()
            obj_val = tf.reduce_sum(obj_val)
            
        # evaluate gradient and update the image parameters
        grad = tape.gradient(obj_val, img_param.param)
        optimizer.apply_gradients([(grad * -1, img_param.param)])

    # evaluate the final images from parameters
    img_batch = img_param.eval()
    
    print(f'Step {max(steps)}')
    display_image_batch(img_batch)
    
    if display_plots:
        max_final_var = var_path[:, -1].max()
        ylim_max = max(max_final_var * 5, 0.005)
        display_plot_batch(var_path, 'mean pixel value changes', ylim=(0, ylim_max))
        display_plot_batch(obj_path, 'objective values')
    
    return img_batch


def display_image_batch(img_batch):
    """Displays a batch of images (in rows of four).

    Parameters
    ----------
    img_batch : 4D tensor
        Batch of images, with dimensions (batch, height, width, channel).
    """

    batch = tf.shape(img_batch).numpy()[0]
    cols, rows = 4, (batch - 1) // 4 + 1

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()
    [ax.axis('off') for ax in axs]
    [axs[i].imshow(img_batch[i].numpy()) for i in range(batch)]
    plt.show()


def display_plot_batch(path_batch, title, **ax_args):
    """Plots a batch of curves (in rows of four).

    Parameters
    ----------
    path_batch : 2D array
        Batch of sequences to be plotted, with dimensions (batch, sequence).
    title : str
        Title of the plots.
    ax_args : keyword arguments
        Axes arguments that are accepted by matplotlib.axes.Axes.set().
    """
    
    batch = path_batch.shape[0]
    cols, rows = 4, (batch - 1) // 4 + 1
    
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(4 * cols, 0.5 + 4 * rows))
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(title)
    axs = axs.flatten()
    for i in range(batch):
        axs[i].plot(path_batch[i])
        axs[i].set(**ax_args)
    for i in range(batch, cols * rows):
        axs[i].axis('off')
    plt.show()
