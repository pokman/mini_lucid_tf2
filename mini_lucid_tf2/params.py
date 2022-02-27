import tensorflow as tf
import numpy as np

# empirical RGB covariance matrix (based on the square root used in Lucid)
RGB_COV = np.array([[0.0761, 0.0692, 0.0627],
                    [0.0692, 0.0754, 0.0714],
                    [0.0627, 0.0714, 0.0819]], dtype=np.float32)


class ImageParam(object):
    """Represents a parametrization of an image (batch) in frequency space. Provides a method to
    evaluate the pixels.

    Attributes
    ----------
    shape : tuple
        Shape of the image batch, of the form (batch size, height, width, 3).
    param : tf.Variable of float
        Parameters of the image batch. At construction, the values are initialized randomly with
        zero mean and specified standard deviation (init_noise).
    freq_decay : float
        Frequency decay rate, controlling the downscaling of high frequency modes when the image
        parameters are evaluated into pixels.
    rgb_corr : bool
        Whether to impose empirical RGB correlations when the image parameters are evaluated into
        pixels.

    Methods
    -------
    eval
        Evaluates the image parameters into pixels.
    """
    
    def __init__(self, h, w=None, batch=1, init_noise=0.01, freq_decay=1.0, rgb_corr=True):
        w = w or h
        ch = 3
        self.shape = (batch, h, w, ch)
        param = np.random.normal(size=self.shape, scale=init_noise)
        self.param = tf.Variable(param, dtype=tf.float32)
        self.freq_decay = freq_decay
        self.rgb_corr = rgb_corr
    
    def eval(self):
        """Evaluates the image parameters into pixels. This consists of several steps, all of which
        are bijections:

        1. Using simple row and column operations, transform the (h, w) matrix of parameters (per
        channel) into a (h, w//2+1) complex-valued matrix that satisfies the exact constraints to be
        the (real) FFT of a (h, w) real-valued matrix.

        2. Rescale values using a (decreasing) function of respective frequencies and the specified
        decay rate (freq_decay).

        3. Apply inverse FFT.

        4. Impose empirical correlations across RGB channels.

        5. Map all values into [0,1] using a sigmoid.

        Returns
        -------
        img_batch : 4D tensor
            Batch of image pixels, of the shape (batch size, height, width, number of channels).
        """

        _, h, w, _ = self.shape
        
        # 'fold' the matrix of raw parameters into a complex-valued matrix
        # in the form of an output of rfft2d
        param_raw = tf.complex(self.param, tf.zeros(self.shape))
        param_raw = tf.transpose(param_raw, perm=[0, 3, 1, 2])
        param_rfft = self._fold_h(h) @ param_raw @ self._fold_w(w)
        
        # scale by (1) a decreasing function of frequencies that maintains the
        # total variance of parameters up to a factor independent of image 
        # size, and (2) another factor that will compensate the variance 
        # scaling in inverse Fourier transform 
        freq_h = np.fft.fftfreq(h)[:, None]
        freq_w = np.fft.fftfreq(w)[None, :w//2+1]
        r = np.sqrt(h * w)
        param_rfft /= (np.sqrt(freq_h ** 2 + freq_w ** 2) + 1 / r) ** self.freq_decay
        param_rfft *= r

        # apply inverse Fourier transform
        param_irfft = tf.signal.irfft2d(param_rfft, fft_length=[h, w])
        param_irfft = tf.transpose(param_irfft, perm=[0, 2, 3, 1])
        
        if self.rgb_corr:
            # impose empirical RGB correlations
            rgb_cov_sqrt = np.linalg.cholesky(RGB_COV)
            max_rgb_sd = np.sqrt(np.diagonal(RGB_COV).max())
            param_irfft = param_irfft @ rgb_cov_sqrt.T / max_rgb_sd
        
        # squeeze into [0,1] (with unit slope at 0)
        img_batch = tf.sigmoid(param_irfft * 4) 
        
        return img_batch
        
    @staticmethod
    def _fold_h(h):
        """Row operations on the matrix of parameters (see eval())."""
        m = np.eye(h, dtype=np.complex64)
        m[1:(h-1)//2+1, 1:(h-1)//2+1] = np.eye((h-1)//2) * (0.5 + 0.5j)
        m[1:(h-1)//2+1, h//2+1:] = np.eye((h-1)//2)[::-1] * (0.5 - 0.5j)
        m[h//2+1:, 1:(h-1)//2+1] = np.eye((h-1)//2)[::-1] * (0.5 - 0.5j)
        m[h//2+1:, h//2+1:] = np.eye((h-1)//2) * (0.5 + 0.5j)
        return m
        
    @staticmethod
    def _fold_w(w):
        """Column operations on the matrix of parameters (see eval())."""
        m = np.eye(w, w//2+1, dtype=np.complex64)
        m[1:(w-1)//2+1, 1:(w-1)//2+1] = np.eye((w-1)//2) * (0.5 + 0.5j)
        m[w//2+1:, 1:(w-1)//2+1] = np.eye((w-1)//2)[::-1] * (0.5 - 0.5j)
        return m
