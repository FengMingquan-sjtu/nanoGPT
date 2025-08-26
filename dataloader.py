import glob
import numpy as np
import torch

# -----------------------------------------------------------------------------
# distributed data loader


class DistributedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.filename = filename
        self.current_position = self.process_rank * self.B * self.T

        if "qwen2" in filename:
            self.data_dtype = np.uint32
        else:
            self.data_dtype = np.uint16


    def next_batch(self):
        B = self.B
        T = self.T
        data = np.memmap(self.filename, dtype=self.data_dtype, mode='r')
        len_data = len(data)
        assert len_data > (B * T * self.num_processes + 1), "Data size is too small for the number of processes and batch size."
        if self.current_position > len_data:
            self.current_position = self.current_position % len_data
        if self.current_position + (B * T + 1) > len_data:
            buf = np.concatenate((data[self.current_position : len_data], data[0 : (self.current_position + B*T + 1) % len_data]))
        else:
            buf = data[self.current_position : self.current_position+B*T+1]
        buf = torch.from_numpy(buf.astype(np.int64))
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        return x, y