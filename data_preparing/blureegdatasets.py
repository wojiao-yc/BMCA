import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import open_clip

os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

# Initialize device (use CUDA if available)
device = "cuda:7" if torch.cuda.is_available() else "cpu"

# Load configuration from YAML file
cfg = OmegaConf.load(os.path.join("/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Get image directories from config
img_directory_training = cfg.eegdataset.img_directory_training
img_directory_test = cfg.eegdataset.img_directory_test


class EEGDataset():
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes=None, pictures=None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200  # Number of classes (1654 for train, 200 for test)
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # Subject to adapt/leave out

        # Verify at least some subjects exist in the directory
        assert any(sub in self.subject_list for sub in self.subjects)

        # Load and process data
        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        
        features_filename = "/home/wenxiao/workspace/qhy/BMCA/data/intra-subject_ubp_EEGProjectLayer_RN50_train.pt" \
            if self.train else "/home/wenxiao/workspace/qhy/BMCA/data/intra-subject_ubp_EEGProjectLayer_RN50_test.pt"

        saved_features = torch.load(features_filename, weights_only=False)

        # ====== Text features ======
        # saved_features['text_features'] 是一个 dict: { word: (1024,) }
        text_dict = saved_features['text_features']

        # 统一排序（保证稳定性）并拼接成矩阵
        text_keys = list(text_dict.keys())
        text_tensor = torch.stack([text_dict[k] for k in text_keys], dim=0)  # shape: [1654, 1024]


        # ====== Image features ======
        # saved_features['img_features']['low'/'medium'/'high'] 都是 dict，同一套 key
        img_low   = saved_features['img_features']['low']
        img_med   = saved_features['img_features']['medium']
        img_high  = saved_features['img_features']['high']

        img_keys = list(img_low.keys())

        # 对应元素求平均
        img_tensor = torch.stack([
            (img_low[k] + img_med[k] + img_high[k]) / 3.0
            for k in img_keys
        ], dim=0)  # shape: [16540, 1024]

        self.text_features = text_tensor
        self.img_features  = img_tensor

        print("Text features shape:", self.text_features.shape)
        print("Image features shape:", self.img_features.shape)

            
    def load_data(self):
        """Load EEG data, labels, text descriptions and image paths"""
        data_list = []
        label_list = []
        texts = []
        images = []
        
        # Determine which image directory to use
        directory = img_directory_training if self.train else img_directory_test
        
        # Get all directories in the path and sort them
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        # Filter directories if specific classes are requested
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

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

        # Handle different cases for image selection
        if self.classes is not None and self.pictures is not None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  # Initialize images list
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            print("Error in image selection parameters")
            
        # Load EEG data from each subject
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        for subject in self.subjects:
            if self.train:
                file_name = 'preprocessed_eeg_training.npy'
                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes = 1654  # Each class contains 10 images
                samples_per_class = 10  # Each class has ten samples
                
                # Handle different data selection cases for training
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                            labels = torch.full((1,), c, dtype=torch.long).detach()
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)
                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
            else:
                # Handle test data
                if subject == self.adap_subject or self.adap_subject == None:  
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']
                    n_classes = 200  # Each class contains 1 image
                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:
                            continue
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                else:
                    continue

        # Process and concatenate all loaded data
        if self.train:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
        label_tensor = torch.cat(label_list, dim=0)
        
        # Additional processing for training labels
        if self.train:
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                    unique_values = torch.tensor(lis)        
                    mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                    label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        self.times = times
        self.ch_names = ch_names

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
        
        # Calculate indices for text and image based on dataset mode
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
                
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
                img_index = (index % index_n_sub_train) // (4)
            else:
                text_index = (index % index_n_sub_test)
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
                
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
                img_index = (index % index_n_sub_train) // (4)
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


if __name__ == "__main__":
    # Example usage
    data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"
    
    # Create datasets
    train_dataset = EEGDataset(data_path, subjects=['sub-01'], train=True)    
    test_dataset = EEGDataset(data_path, subjects=['sub-01'], train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test sample access
    i = 80*1-1
    x, label, text, text_features, img, img_features = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)