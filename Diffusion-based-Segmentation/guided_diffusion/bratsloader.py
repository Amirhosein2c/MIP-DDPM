import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            return (image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label)

    def __len__(self):
        return len(self.database)


class PSMADataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        
        print(f"test flag: {self.test_flag}")
        
        if test_flag:
            self.seqtypes = ['PET', 'PET10', 'PET5', 'PETD']
            print("Here!")
        else:
            self.seqtypes = ['PET', 'PET10', 'PET5', 'PETD', 'SEG']

        self.seqtypes_set = set(self.seqtypes)
        print(f"seqtypes_set: {self.seqtypes_set}")
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    if f.find("SEG") > 0 and test_flag:
                        pass
                    else:
                        seqtype = f.split('_')[1]
                        datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                asghar = "agha"

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            out = torch.nn.functional.pad(out, pad=(3,3,3,3), mode="constant")
            image = out
            # print(f"{image.shape}")
            return (image, path)
        else:
            out = torch.nn.functional.pad(out, pad=(3,3,3,3), mode="constant")
            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label)

    def __len__(self):
        return len(self.database)

