import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import open_clip

# Get image directories from config

class EEGDataset():
    def __init__(self, data_path, subjects=None, train=True, avg=True ,time_window=[0, 1.0]):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200  # Number of classes (1654 for train, 200 for test)
        self.avg = avg

        # Verify at least some subjects exist in the directory
        assert any(sub in self.subject_list for sub in self.subjects)

        # Load and process data
        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        
        # Load precomputed CLIP features
        features_filename = "/home/wenxiao/workspace/qhy/BMCA/data/intra-subject_ubp_EEGProjectLayer_RN50_train.pt" \
            if self.train else "/home/wenxiao/workspace/qhy/BMCA/data/intra-subject_ubp_EEGProjectLayer_RN50_test.pt"
        # features_filename = "/home/wenxiao/workspace/qhy/BMCA/data/RN50_openai_train.pt" \
        #     if self.train else "/home/wenxiao/workspace/qhy/BMCA/data/RN50_openai_test.pt"
        print("Loading features from:", features_filename)
        saved_features = torch.load(features_filename, weights_only=False)
        # # Text features
        text_dict = saved_features['text_features']
        text_keys = list(text_dict.keys())
        text_tensor = torch.stack([text_dict[k] for k in text_keys], dim=0)  # shape: [1654, 1024]
        # Img features
        img_low   = saved_features['img_features']['low']
        img_med   = saved_features['img_features']['medium']
        img_high  = saved_features['img_features']['high']
        img_keys = list(img_low.keys())
        img_tensor = torch.stack([
            (img_low[k] + img_med[k] + img_high[k]) / 3.0
            # img_med[k]
            for k in img_keys
        ], dim=0)  # shape: [16540, 1024]
        # img_tensor = saved_features['img_features']
        # img_keys = list(img_tensor.keys())
        # img_tensor = torch.stack([img_tensor[k] for k in img_keys], dim=0)  # shape: [16540, 1024]
        self.text_features = text_tensor
        self.img_features  = img_tensor
        # self.text_features = saved_features['text_features']
        # self.img_features = saved_features['img_features']
        print("Text features shape:", self.text_features.shape)
        print("Image features shape:", self.img_features.shape)

            
    def load_data(self):
        """Load EEG data, labels, text descriptions and image paths"""
        data_list = []
        label_list = []
        texts = []
        images = []
        
        # Determine which image directory to use
        img_directory_training = "/mnt/dataset1/ldy/4090_Workspace/4090_THINGS/images_set/training_images"
        img_directory_test = "/mnt/dataset1/ldy/4090_Workspace/4090_THINGS/images_set/test_images"
        directory = img_directory_training if self.train else img_directory_test
        
        # Get all directories in the path and sort them
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        # Extract text descriptions from directory names
        for dir in dirnames:
            try:
                idx = dir.index('_')
                description = dir[idx+1:]  # Get content after first '_'
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
            new_description = f"This picture is {description}"
            texts.append(new_description)

        # Determine image directory based on train/test mode
        img_directory = img_directory_training if self.train else img_directory_test
        # Get all image folders and sort them
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        images = []  # Initialize images list
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)

            
        # Load EEG data from each subject
        print("self.subjects", self.subjects)
        selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        for subject in self.subjects:
            if self.train:
                file_name = 'preprocessed_eeg_training.npy'
                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()

                # data_path = os.path.join("/mnt/dataset4/qhy/THINGS-EEG/things_eeg/Preprocessed_data_250Hz_whiten", subject, 'train.pt')
                # loaded_data = torch.load(data_path, weights_only=False)
                # preprocessed_eeg_data=torch.from_numpy(loaded_data['eeg']).float().detach()
                
                              
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes = 1654  # Each class contains 10 images
                samples_per_class = 10  # Each class has ten samples
                
                for i in range(n_classes):
                    start_index = i * samples_per_class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)
            else:
                file_name = 'preprocessed_eeg_test.npy'
                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)

                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                # data_path = os.path.join("/mnt/dataset4/qhy/THINGS-EEG/things_eeg/Preprocessed_data_250Hz_whiten", subject, 'test.pt')
                # loaded_data = torch.load(data_path, weights_only=False)
                # preprocessed_eeg_data=torch.from_numpy(loaded_data['eeg']).float().detach()

                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']
                n_classes = 200  # Each class contains 1 image
                samples_per_class = 1

                for i in range(n_classes):
                    start_index = i * samples_per_class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                    preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)

        # Process and concatenate all loaded data
        label_tensor = torch.cat(label_list, dim=0)
        if self.train and not self.avg:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
            label_tensor = label_tensor.repeat_interleave(4)
        elif self.train and self.avg:
            data_tensor = torch.cat(data_list, dim=0).view(-1, 4, *data_list[0].shape[2:]).mean(dim=1)              
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
        
    
        self.times = times
        self.ch_names = ch_names

        selected_idx = [self.ch_names.index(ch) for ch in selected_ch]
        data_tensor = data_tensor[:,selected_idx,:]
        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")

        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):
        """
        Extract EEG data within specified time window
        
        Args:
            eeg_data: Raw EEG data tensor
            time_window: [start, end] time window in seconds
            
        Returns:
            Extracted EEG data within the time window
        """
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]
        return extracted_data
    
    def __getitem__(self, index):
        """
        Get a single data sample by index
        
        Returns:
            x: EEG data
            label: Class label
            text: Text description
            text_features: CLIP text features
            img: Image path
            img_features: CLIP image features
        """
        x = self.data[index]
        label = self.labels[index]
        
        if self.avg:
            index_n_sub_train = self.n_cls * 10 * 1
        else:
            index_n_sub_train = self.n_cls * 10 * 4
        index_n_sub_test = self.n_cls * 1 * 80
                
        if self.train and not self.avg:
            text_index = (index % index_n_sub_train) // (10 * 4)
            img_index = (index % index_n_sub_train) // (4)
        elif self.train and self.avg:
            text_index = (index % index_n_sub_train) // (10 * 1)
            img_index = (index % index_n_sub_train) // (1)
        else:
            text_index = (index % index_n_sub_test)
            img_index = (index % index_n_sub_test)
                
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return x, label, text, text_features, img, img_features

    def __len__(self):
        """Return total number of samples in dataset"""
        return self.data.shape[0]