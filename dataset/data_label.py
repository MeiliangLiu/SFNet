import os
import numpy as np
# from tqdm import tqdm_notebook
# import cnn_interpretability.utils as utils
from torch.utils.data import Dataset, DataLoader
import torch
import nibabel as nib
from scipy.ndimage.interpolation import zoom
img_dir = "./AD_class"
# 返回一个列表，包含每个nii文件的路径
# def set_filenames():
#     # Set filenames
#     filenames = filter(lambda filename: filename.endswith('nii.gz'), os.listdir(img_dir))
#     filenames = [os.path.join(img_dir, filename) for filename in filenames]
#     return filenames

def set_filenames():
    ad_dir = os.path.join(img_dir, "AD")
    cn_dir = os.path.join(img_dir, "CN")

    ad_filenames = [os.path.join(ad_dir, filename) for filename in os.listdir(ad_dir) if filename.endswith("nii.gz")]
    cn_filenames = [os.path.join(cn_dir, filename) for filename in os.listdir(cn_dir) if filename.endswith("nii.gz")]

    return ad_filenames, cn_filenames


# 返回一个标签数组
def set_labels(filenames):
    # AD_TEST
    labels = map(lambda filename: 1 if filename[21:23] == 'AD' else 0, filenames)
    labels_arr = []
    for x in labels:
        labels_arr.append(x)
    labels = np.array(labels_arr)[:, None]
    return labels


def load_nifti(file_path, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_fdata())
    # print(struct_arr.shape)

    if remove_nan:
        # struct_arr = np.nan_to_num(struct_arr).astype(np.float32)
        struct_arr = np.nan_to_num(struct_arr)

    return struct_arr



# 返回处理后的数据
class ADNIDataset(Dataset):


    def __init__(self, filenames, labels,transform=None,mean=None, std=None, normalize=True):
        self.filenames = filenames
        self.labels = torch.LongTensor(labels)
        self.transform=transform
        # self.normalize = normalize
        # self.mean = mean#整个训练集的均值
        # self.std = std

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        label = self.labels[idx]

        struct_arr = load_nifti(self.filenames[idx])

        struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)  # prevent 0 division by adding small factor
        if self.transform:
            struct_arr = self.transform(struct_arr)
        struct_arr = struct_arr[None]  # add (empty) channel dimension
        struct_arr = torch.FloatTensor(struct_arr)


        return struct_arr, label

    def image_shape(self):
        """The shape of the MRI images."""
        return load_nifti(self.filenames[0]).shape


# 分层抽样
class StratifiedSampler(torch.utils.data.Sampler):

    def __init__(self, class_vector, batch_size):

        # class_vector 类别标签的向量，得出分层抽样的分割数
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')

        cn = []
        ad = []
        # smci=[]

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.25, random_state=19)
        X = torch.randn(self.class_vector.size(0), 4).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        indices = np.hstack([train_index, test_index])
        # 把索引按照类别分类
        for i in indices:
            if y[indices[i]] == 0:
                cn.append(indices[i])
            elif y[indices[i]] == 1:
                ad.append(indices[i])
            # elif y[indices[i]] == 2:
            #     smci.append(indices[i])


        # 保证新索引中每个类别数量均衡
        new_indices = []
        # min_length = min(len(cn), len(ad))
        min_length = min(len(cn), len(ad))

        for i in range(min_length):
            new_indices.append(cn[i])
            new_indices.append(ad[i])
            # new_indices.append(smci[i])

        return new_indices

    # 返回生成的索引
    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)





