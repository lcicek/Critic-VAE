from cv2 import log
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

from parameters import SAVE_IMAGES, SAVE_PATH, LOSS, log_count
from utility import prepare_rgb_image

class Logger(object):

    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        #summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def image_summary(self, tag, images, epoch, labels=None):
        """Log a list of images."""

        for i in range(log_count):
            rec = images[0][i] # reconstructed image
            orig = images[1][i] # original image
            
            conc_h = np.concatenate((orig, rec), axis=2) # concatenate horizontally
            img_array, img = prepare_rgb_image(conc_h)

            # save image locally
            if SAVE_IMAGES:
                img.save(f'{SAVE_PATH}/img{i}-EP{epoch}-L{labels[i]}.png', format="png")

            if labels is None:
                self.writer.add_image(tag='img%d-ep%d/%s' % (i, epoch, LOSS), img_tensor=img_array, dataformats='HWC')
            else:
                self.writer.add_image(tag='img%d-ep%d-label=%d/%s' % (i, epoch, labels[i], LOSS), img_tensor=img_array, dataformats='HWC')

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

        # Create a histogram using numpy
        #counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        #hist = tf.compat.v1.HistogramProto()
        #hist.min = float(np.min(values))
        #hist.max = float(np.max(values))
        #hist.num = int(np.prod(values.shape))
        #hist.sum = float(np.sum(values))
        #hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        #bin_edges = bin_edges[1:]

        # Add bin edges and counts
        #for edge in bin_edges:
        #    hist.bucket_limit.append(edge)
        #for c in counts:
        #    hist.bucket.append(c)

        # Create and write Summary
        #summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        #self.writer.add_summary(summary, step)
        #self.writer.flush()