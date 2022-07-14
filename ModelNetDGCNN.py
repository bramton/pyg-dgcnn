import os
from pathlib import Path
import torch
import h5py

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data

# Original: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py
class ModelNetDGCNN(InMemoryDataset):
    """ This is the dataset used by the original authors of the Dynamic Graph
    CNN for Learning on Point Clouds paper. For an unknown reason, the training
    dataset has three less samples compared to the original ModelNet40 dataset
    (9840 vs 9843). This dataset comes pre-processed. Most likely, the models
    have been centered and normalised to fit a unit sphere.
    """
    url='https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

    def __init__(self, root, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return ['ply_data_test0.h5','ply_data_test1.h5',
            'ply_data_train0.h5','ply_data_train1.h5',
            'ply_data_train2.h5','ply_data_train3.h5',
            'ply_data_train4.h5']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process_set(self, dataset):
        data_list = []
        files = [f for f in self.raw_file_names if dataset in f]
        for f in files:
            with h5py.File(Path(self.raw_dir)/f, 'r') as f:
                for i in range(len(f['data'])):
                    data = Data()
                    data.pos = torch.tensor(f['data'][i].astype('float32'))
                    data.y = torch.tensor(f['label'][i][0]).type(torch.long)
                    data_list.append(data)

        return self.collate(data_list)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = Path(self.root)/'modelnet40_ply_hdf5_2048'
        os.rename(folder, self.raw_dir)
