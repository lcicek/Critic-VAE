# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
#from tensorflow.compat import v1 as tf # new tensorflow version doesnt have below functions
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from io import BytesIO
from PIL import Image

#tf.disable_eager_execution()


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        #summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def image_summary(self, tag, images, step, labels=None):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            #try:
            #    s = StringIO()
            #except:
            s = BytesIO()
                
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img, mode='RGB').save(s, format="png")

            self.writer.add_image(tag='%s/%d' % (tag, i), img_tensor=img, dataformats='CHW')

            # Create an Image object
            #img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
            #                           height=img.shape[0],
            #                           width=img.shape[1])
            # Create a Summary value
            #if labels is None:
            #    img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
            #else:
            #    img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d/%d' % (tag, labels[i], i), image=img_sum))

        # Create and write Summary
        #summary = tf.compat.v1.Summary(value=img_summaries)
        #self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(input=values, bins=bins)

"""
class Logger2(object):
    
    def __init__(self, log_dir):
        '''Create a summary writer logging to log_dir.'''
        # tf.disable_eager_execution()
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        '''Log a scalar variable.'''
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        '''Log a list of images.'''

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()

            img = Image.fromarray(np.uint8(img))
            img.save(s, format='png')

            width, height = img.size
            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=height,
                                       width=width)
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        
    def histo_summary(self, tag, values, step, bins=1000):
        '''Log a histogram of the tensor of values.'''

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
"""