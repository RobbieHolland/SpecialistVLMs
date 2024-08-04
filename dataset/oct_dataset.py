import os
import imageio
import numpy as np
import pandas as pd
import torch
from dataset.image_transforms import center_crop, center_crop_and_resize, flip_horizontal, com_crop
from datetime import datetime
from tqdm import tqdm

def image_preprocessing(images, crop_size, scale_size, flip_right):
    if flip_right:
        images = flip_horizontal(images)
    elif (crop_size is not None) and (scale_size is None):
        images = center_crop(images, size=crop_size)
    elif (crop_size is not None) and (scale_size is not None):
        images = center_crop_and_resize(images, crop_size=crop_size, final_size=scale_size)
    return images

surface_names = ['ILM             ', 'RNFL-GCL        ', 'GCL-IPL         ',
                 'IPL-INL         ', 'INL-OPL         ', 'OPL-HFL         ',
                 'BMEIS           ', 'IS/OSJ          ', 'IB_OPR          ',
                 'IB_RPE          ', 'OB_RPE          ',] # "Bruch's Membrane",]

class OCTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 params,
                 set_name,
                 data_csv,
                 image_column,
                 transform,
                 target_transform,
                 seed=None):

        self.params = params
        self.set_name = set_name
        self.image_columns = [image_column]
        self.image_transform = transform
        self.image_dir = params['image_dir']
        self.target_columns = [params['target']]
        self.target_transform = target_transform
        self.target_is_categorical = [False]
        self.crop_size = params['crop_size']
        self.scale_size = params['scale_size']
        self.number_train_labels = params['number_train_labels']

        # Maybe load data CSV
        if isinstance(data_csv, pd.DataFrame):
            self.data_csv_name = '?'
            self.data_csv = data_csv
        else:
            self.data_csv_name = data_csv
            if data_csv.split('.')[-1] == 'pkl':
                self.data_csv = pd.read_pickle(data_csv).reset_index(drop=True)
            elif data_csv.split('.')[-1] == 'csv':
                self.data_csv = pd.read_csv(data_csv).reset_index(drop=True)
            else:
                raise Exception(f'Unknown type for dataset metadata: {data_csv}')

        self.data_csv['include'] = True

        if self.params['sort_by_date']:
            self.data_csv = self.data_csv.sort_values('UnixDays').reset_index(drop=True)

        # Preloading images
        self.images = None
        if params['preload_images']:
            print('Preloading dataset of images...')
            before = datetime.now()

            all_images = []
            for _, image_name in tqdm(self.data_csv[self.image_columns[0]].items()):
                images = self._get_images(image_name=image_name, transform=False)
                all_images += [images]
            self.images = np.stack(all_images)
            print('...took', datetime.now() - before)

    def filter_dataset(self, filter_, target):
        na_filter = lambda obj: obj.notna().any(axis=1) if isinstance(obj, pd.DataFrame) else obj.notna()

        f = na_filter
        if filter_ is not None:
            f = lambda df: na_filter(df) & filter_(df)

        return self.data_csv.loc[f(self.data_csv[target])].reset_index(drop=True)

    def group_dataset_by(self, grouped_by):
        df = self.data_csv
        grouped_df_gen = df.groupby(grouped_by)

        import numbers
        to_list = lambda l: np.array(l) if all([isinstance(x, numbers.Number) for x in l]) else list(l)

        grouped_df = grouped_df_gen[self.image_columns].apply(to_list).reset_index()

        for col in list(set(df.columns) - set(grouped_df)):
            grouped_df[col] = grouped_df_gen[col].apply(to_list).reset_index()[col]

        grouped_df['Length'] = grouped_df['ImageId'].apply(lambda ims: len(ims))
        grouped_df['GroupID'] = grouped_df.index

        grouped_df['DurationYears'] = grouped_df['UnixYears'].apply(lambda t: t.max() - t.min())

        return grouped_df

    def _get_images(self, index=0, image_name=None, transform=True, central_crop=False, flip_right=False):
        images = []
        random_crop_params = None

        if self.images is not None:
            images = [self.images[index].copy()]
        else:
            for image_number in range(len(self.image_columns)):
                if image_name is None:
                    image_name = self.data_csv.loc[index, self.image_columns[image_number]]
                image_name = image_name.replace('.png', '.' + self.params['extension'])
                
                imagepath = os.path.join(self.image_dir, image_name)
                suffix = os.path.splitext(imagepath)[1]
                if suffix == '.png':
                    images.append(np.expand_dims(np.array(imageio.imread(imagepath)), 0))
                elif suffix == '.bsdf':
                    images.append(np.transpose(imageio.volread(imagepath), (1, 0, 2)))
                elif suffix == '.npy':
                    images.append(np.load(imagepath))
                else:
                    raise Exception

        # Assume one image
        images = images[0]

        if not transform:
            return images

        images = images / 255

        # Strip black space from image
        images = image_preprocessing(images, (np.array(self.crop_size) * self.params['image_scale']).tolist(), self.scale_size, flip_right)
        if central_crop:
            return images

        images = np.expand_dims(images, 1)

        before = datetime.now()
        if self.image_transform is not None:
            t = self.image_transform(images)

            if self.target_columns[0][0] in surface_names:
                images = torch.Tensor(t['image']).unsqueeze(0)
                random_crop_params = [s['params'] for s in t['replay']['transforms'] if 'RandomCrop' in s['__class_fullname__']]
                if random_crop_params == []:
                    random_crop_params = np.array([0.5,0.5])
                else:
                    random_crop_params = np.array(list(random_crop_params[0].values()))
                
            else:
                images = torch.Tensor(t['image']).unsqueeze(0)

        return images[0], random_crop_params

    def _get_targets(self, index, random_crop_params=None):
        targets = []
        for target_number in range(len(self.target_columns[0])):
            target_column = self.target_columns[0][target_number]

            if target_column in surface_names:  # Loading segmentation
                layer_ix = surface_names.index(target_column)

                target = np.array(self.segmentations[index, :, layer_ix])
                targets.append(target)

            else:  # Loading target straight from CSV
                target = self.data_csv.loc[index, target_column]
                if isinstance(target, str):
                    targets.append(np.array(target))
                else:
                    targets.append(np.array(target).astype(np.float32))

        if self.target_transform is not None:
            targets = self.target_transform(targets, random_crop_params)

        return targets

    def __len__(self):
        return len(self.data_csv)

class DatasetFromCSV(OCTDataset):
    def __init__(self,
                 params,
                 set_name,
                 data_csv,
                 image_column, transform,
                 target_transform,
                 longitude_samples=None,
                 seed=None):
        super().__init__(params, set_name, data_csv, image_column, transform, target_transform, seed)

    def __getitem__(self, index):

        images, random_crop_params = self._get_images(index)
        targets = self._get_targets(index, random_crop_params=random_crop_params)

        return images, targets
