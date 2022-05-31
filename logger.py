from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step,  new_style=True)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(input=values, bins=bins)

    """
    def image_summary(self, tag, images, epoch, labels=None):

        for i in range(log_count):
            rec = images[0][i] # reconstructed image
            orig = images[1][i] # original image
            
            conc_h = np.concatenate((orig, rec), axis=2) # concatenate horizontally
            img_array, img = prepare_rgb_image(conc_h)

            if labels is None:
                self.writer.add_image(tag='img%d-ep%d' % (i, epoch), img_tensor=img_array, dataformats='HWC')
            else:
                self.writer.add_image(tag='img%d-ep%d-label=%d' % (i, epoch, labels[i]), img_tensor=img_array, dataformats='HWC')
    """