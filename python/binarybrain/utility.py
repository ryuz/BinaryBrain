

from binarybrain import binarybrain as bb
from tqdm import tqdm


def calculation(net, x, x_shape, t, t_shape, max_batch_size, min_batch_size=1,
            metrics=None, loss=None, optimizer=None, train=False,
            print_loss=True, print_metrics=True):
    
    if metrics is not None:
        metrics.clear()
    
    if loss is not None:
        loss.clear()
    
    batch_size = len(x)
    
    x_buf = bb.FrameBuffer()
    t_buf = bb.FrameBuffer()
    
    for index in tqdm(range(0, batch_size, max_batch_size)):
        # calc mini_batch_size
        mini_batch_size = min(max_batch_size, batch_size-index)
        
        # setup x
        x_buf.resize(bb.TYPE_FP32, mini_batch_size, x_shape)
        x_buf.set_data(x[index:index+mini_batch_size])
        
        # forward
        y_buf = net.forward(x_buf, train)
        
        # setup t
        t_buf.resize(bb.TYPE_FP32, mini_batch_size, t_shape)
        t_buf.set_data(t[index:index+mini_batch_size])
        
        # calc loss
        if loss is not None:
            dy_buf = loss.calculate_loss(y_buf, t_buf, mini_batch_size)

        # calc metrics
        if metrics is not None:
            metrics.calculate_metrics(y_buf, t_buf)

        # backward
        if train and loss is not None:
            net.backward(dy_buf)

            # update
            if  optimizer is not None:
                optimizer.update()


class Runner:
    def __init__(
            self,
            net,
            name="",
            loss=None,
            metrics=None,
            optimizer=None,
            max_run_size=0,
            print_progress=True,
            print_progress_loss=True,
            print_progress_accuracy=True,
            log_write=True,
            log_append=True,
            file_read=False,
            file_write=False,
            write_serial=False,
            initial_evaluation=False,
            seed=1):
        self.net                     = net
        self.name                    = name
        self.loss                    = loss
        self.metrics                 = metrics
        self.optimizer               = optimizer
        self.max_run_size            = max_run_size
        self.print_progress          = print_progress
        self.print_progress_loss     = print_progress_loss
        self.print_progress_accuracy = print_progress_accuracy
        self.log_write               = log_write
        self.log_append              = log_append
        self.file_read               = file_read
        self.file_write              = file_write
        self.write_serial            = write_serial
        self.initial_evaluation      = initial_evaluation
    
    def fitting(self, td, epoch_size, mini_batch_size=16):
        for epoch in range(epoch_size):
            # train
            calculation(self.net, td.x_train, td.x_shape, td.t_train, td.t_shape, mini_batch_size, mini_batch_size,
                        self.metrics, self.loss, self.optimizer, train=True, print_loss=True, print_metrics=True)

            # evaluation
            calculation(self.net, td.x_test, td.x_shape, td.t_test, td.t_shape, mini_batch_size, 1, self.metrics, self.loss)
            print('epoch=%d metrics=%f  loss=%f' % (epoch, self.metrics.get_metrics(), self.loss.get_loss()))

