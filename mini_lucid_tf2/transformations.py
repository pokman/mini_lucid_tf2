import tensorflow as tf
import numpy as np


def pad(n):
    """Constructs a tensor-native function that pads a batch of images with n extra pixels on each
    side, i.e. it increases dimensions by (0, 2n, 2n, 0)."""
    def tr(img_batch):
        paddings = [[0, 0], [n, n], [n, n], [0, 0]]
        img_batch = tf.pad(img_batch, paddings, mode='CONSTANT', constant_values=0.5)
        return img_batch
    return tr


def random_crop(n):
    """Constructs a tensor-native function that strips from a batch of images a randomly selected
    border consisting of n pixels in both vertical and horizontal directions, i.e. it reduces
    dimensions by (0, n, n, 0)."""
    def tr(img_batch):
        crop_shape = tf.shape(img_batch) - [0, n, n, 0]
        img_batch = tf.image.random_crop(img_batch, crop_shape)
        return img_batch
    return tr


def random_scale(min_scale, max_scale):
    """Constructs a tensor-native function that scales a batch of images by a factor randomly
    selected between min_scale and max_scale."""
    def tr(img_batch):
        scale = np.random.uniform(min_scale, max_scale)
        new_shape = tf.cast(tf.cast(tf.shape(img_batch)[1:3], tf.float32) * scale, tf.int32)
        img_batch = tf.image.resize(img_batch, new_shape)
        return img_batch
    return tr


def random_rotate(max_angle, p=1.0):
    """Constructs a tensor-native function that rotates a batch of images by an angle randomly
    selected, either uniformly between -max_angle and max_angle with probability p, or to be 0 with
    probability 1-p."""
    def tr(img_batch):
        if np.random.rand() > p:
            return img_batch
        angle = np.random.uniform(-max_angle, max_angle)
        img_batch = _rotate(img_batch, angle)
        return img_batch
    return tr


def _rotate(img_batch, angle, fill=0.5, interpolate='bilinear'):
    
    """Rotates a batch of images by an angle, in a tensor-native way.

    Note: The author decides to implement a rotation function 'from scratch' because (1) the
    rotation function used in Lucid seems to be deprecated in tensorflow 2, and (2) the only other
    tensor-native rotation function the author is aware of is in tensorflow-addons, but the author
    wants to avoid dependency on another package.
    
    Parameters
    ----------
    img_batch : 4D tensor
        Batch of images, with dimensions (batch, height, width, channel).
    angle : float
        Angle of rotation.
    fill : float or int or 'nearest'
        Method of filling the exterior. In the case of a float or int, the exterior is filled with
        the given constant value. In the case of 'nearest', the exterior is filled according to the
        nearest pixel on the boundary.
    interpolate : 'nearest' or 'bilinear'
        Method of interpolation between rotated grid points. In the case of 'nearest', every point
        is interpolated to the nearest rotated grid point. In the case of 'bilinear', every point is
        interpolated bilinearly from the four neighboring grid points.
    
    Returns
    -------
    img_batch : 4D tensor
        Batch of rotated images, with the same dimensions.
    """
    
    if type(fill) in [float, int]:
        # pad with an extra pixel around the boundary with the specified value
        img_batch = tf.pad(img_batch, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=fill)
    elif fill != 'nearest':
        raise Exception("invalid value of fill")
    
    # start with the original spatial indices
    batch, h, w, ch = tf.shape(img_batch).numpy()
    idx = np.array([[[i, j] for j in range(w)] for i in range(h)])
    
    # rotate indices by the specified angle (corresponding to rotating pixels
    # by the same angle in the other direction)
    center = [(h-1)/2., (w-1)/2.]
    angle *= np.pi / 180
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    idx_rot = (idx - center) @ rot + center
    
    # move rotated indices outside the boundary to the nearest point on the 
    # boundary (corresponding to filling pixels outside the rotated image
    # according to the nearest point on the rotated boundary)
    #
    # Note: The chosen ceilings ensure that all 4 neighboring grid points to
    # appear below are within the range [0:h, 0:w].
    #
    idx_rot = np.minimum(np.maximum(idx_rot, [0, 0]), [h-1.0001, w-1.0001]) 

    if interpolate == 'nearest':
        # round off rotated indices to the nearest grid point (corresponding to 
        # interpolating pixels to the nearest rotated grid point)
        idx_rot = np.rint(idx_rot).astype(np.int32)
        idx_rot = _add_batch_channel_idx(idx_rot, batch, ch)
        # evaluate the rotated image
        img_rot = tf.gather_nd(img_batch, idx_rot)
    elif interpolate == 'bilinear':
        # replace rotated indices by the 4 neighboring grid points and their
        # weights (corresponding to interpolating pixels bilinearly from the
        # 4 neighboring rotated grid points)
        idx_rot_grid, wts = np.zeros((h, w, 2, 2, 2), dtype=np.int32), np.zeros((h, w, 2, 2))
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            idx_rot_grid[:, :, i, j, :] = np.floor(idx_rot) + [i, j]
            wts[:, :, 1-i, 1-j] = np.prod(idx_rot - idx_rot_grid[:, :, i, j, :], axis=-1) * (-1) ** (i+j)
        idx_rot_grid = _add_batch_channel_idx(idx_rot_grid, batch, ch)
        # evaluate the rotated image
        img_rot_grid = tf.gather_nd(img_batch, idx_rot_grid)
        img_rot = tf.reduce_sum(img_rot_grid * np.expand_dims(wts, axis=(0, 3)), axis=(-2, -1))
    else:
        raise ValueError("interpolate must be 'bilinear' or 'nearest'")

    if type(fill) in [float, int]:
        # restore the original shape
        img_rot = img_rot[:, 1:-1, 1:-1, :]
    
    return img_rot


def _add_batch_channel_idx(idx, batch, ch):
    """Expands an array of spatial indices into an array of indices that also
    include batch and channel dimensions. 
    
    The input idx is expected to be an array of shape (h, w, ..., 2) with, say,
    idx[i, j, ...] = [i', j']. 
    
    The output will then be an array of shape (batch, h, w, ch, ..., 4) with
    idx[b, i, j, c, ...] = [b, i', j', c].
    """
    sh = idx.shape[:-1]
    idx = [np.append(np.ones((*sh, 1)) * b, idx, axis=-1) for b in range(batch)]
    idx = np.stack(idx, axis=0)
    idx = [np.append(idx, np.ones((batch, *sh, 1)) * c, axis=-1) for c in range(ch)]
    idx = np.stack(idx, axis=3)
    return idx.astype(np.int32)


def default_list():
    """Default list of transformations (as in Lucid)."""
    l = [pad(12), 
         random_crop(8), 
         random_scale(0.9, 1.1), 
         random_rotate(10, p=0.8), 
         random_crop(4)]
    return l
