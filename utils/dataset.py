import lmdb
from PIL import Image
from io import BytesIO
from imutils.paths import list_files
from torch.utils.data import Dataset
from torchvision import transforms as TF


def init_transform(args):
    transform = []

    if not args.dataset_type == 'prepared':
        transform.append(TF.Resize((args.resolution, args.resolution)))

    transform.append(TF.RandomHorizontalFlip())
    transform.append(TF.ToTensor())
    transform.append(TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = TF.Compose(transform)
    return transform


def init_dataset(args):
    if args.dataset_type == 'lmdb':
        datatype = LMDBDataset
    elif args.dataset_type == 'normal':
        datatype = NormalDataset
    elif args.dataset_type == 'prepared':
        datatype = PreparedDataset
    else:
        raise NotImplementedError

    transform = init_transform(args)
    return datatype(args, transform)


class LMDBDataset(Dataset):
    def __init__(self, args, transform):
        self.env = lmdb.open(
            args.dataset_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', args.dataset_path)

        self.keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for idx, (key, _) in enumerate(cursor):
                self.keys.append(key)

        self.length = len(self.keys)
        self.resolution = args.resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).resize((self.resolution, self.resolution))
        img = self.transform(img)

        return img


class PreparedDataset(Dataset):
    def __init__(self, args, transform):
        self.env = lmdb.open(
            args.dataset_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', args.dataset_path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.length = self.length
        self.resolution = args.resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


IMG_EXTENSIONS = ['webp', '.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']


class NormalDataset(Dataset):
    def __init__(self, args, transform):
        self.files = []
        listed_files = sorted(list(list_files(args.dataset_path)))

        self.resolution = args.resolution
        self.transform = transform
        self.length = len(listed_files)

        for i in range(self.length):
            file = listed_files[i]
            if any(file.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                self.files.append(file)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = self.transform(img)

        return img
