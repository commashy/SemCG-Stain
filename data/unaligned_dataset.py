import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
random.seed(523)
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # Extract class labels from filenames
        A_class_label = self.get_class_label(A_path)
        B_class_label = self.get_class_label(B_path)

        # Convert images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
            'A_class': A_class_label,
            'B_class': B_class_label
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_class_label(self, path):
        # Extract the class identifier from the filename
        filename = os.path.basename(path)
        # Example filename: 'trainA_05_patch_0.jpg'
        # Split and extract the part after 'trainA_' and before '_patch'
        class_id_str = filename.split('_')[1]
        # Mapping of class identifiers
        class_mapping = {
            'l1': 0, # normal samples
            'l2': 0,
            '05': 1, # acinar samples
            'y2': 1,
            '24': 2, # papillary samples
            '54': 2,
            '33': 3, # solid samples
            '53': 3,
            'y3': 4, # Lepidic, Acinar, Fibrosis sample
            's1': 5, # Lepidic samples
            '06': 6, # Micro-papillary samples
            '52': 7, # Acinar, Micro-papillary, Lepidic
        }
        # class_mapping = {
        #     'l1': 0, 
        #     'l2': 0,
        #     's1': 1,
        #     'y2': 2,
        #     'y3': 3,
        #     '05': 4, 
        #     '06': 5, 
        #     '24': 6, 
        #     '33': 7,
        #     '52': 8, 
        #     '54': 9,
        # }
        # Add your actual mappings for all classes
        class_label = class_mapping.get(class_id_str, -1)

        if class_label == -1:
            raise ValueError(f"Unknown class identifier '{class_id_str}' in filename '{filename}'")
            # Generate a fake class label if the identifier is not available
            # class_label = random.randint(0, len(class_mapping) - 1)
        return class_label
