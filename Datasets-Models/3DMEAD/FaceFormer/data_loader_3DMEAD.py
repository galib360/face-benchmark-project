# import os
# import torch
# from collections import defaultdict
# from torch.utils import data
# import copy
# import numpy as np
# import pickle
# from tqdm import tqdm
# import random,math
# from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
# import librosa    

# class Dataset(data.Dataset):
#     """Custom data.Dataset compatible with data.DataLoader."""
#     def __init__(self, data,subjects_dict,data_type="train"):
#         self.data = data
#         self.len = len(self.data)
#         self.subjects_dict = subjects_dict
#         self.data_type = data_type
#         self.one_hot_labels = np.eye(len(subjects_dict["train"]))

#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         # seq_len, fea_dim
#         file_name = self.data[index]["name"]
#         audio = self.data[index]["audio"]
#         vertice = self.data[index]["vertice"]
#         template = self.data[index]["template"]
#         sentence_id = int(file_name.split(".")[0].split("_")[1])
#         emotion_id = int(file_name.split(".")[0].split("_")[2])
#         intensity_id = int(file_name.split(".")[0].split("_")[3])
#         if self.data_type == "train":
#             subject = file_name.split("_")[0]
#             one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
#         else:
#             one_hot = self.one_hot_labels
#         return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

#     def __len__(self):
#         return self.len

# def read_data(args):
#     print("Loading data...")
#     data = defaultdict(dict)
#     train_data = []
#     valid_data = []
#     test_data = []

#     audio_path = os.path.join(args.dataset, args.wav_path)
#     vertices_path = os.path.join(args.dataset, args.vertices_path)
#     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

#     template_file = os.path.join(args.dataset, args.template_file)
#     with open(template_file, 'rb') as fin:
#         templates = pickle.load(fin,encoding='latin1')
    
#     for r, ds, fs in os.walk(audio_path):
#         for f in tqdm(fs):
#             if f.endswith("wav"):
#                 wav_path = os.path.join(r,f)
#                 speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
#                 input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
#                 key = f.replace("wav", "npy")
#                 data[key]["audio"] = input_values
#                 subject_id = key.split("_")[0]                
#                 temp = templates[subject_id]
#                 data[key]["name"] = f
#                 data[key]["template"] = temp.reshape((-1)) 
#                 vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
#                 if not os.path.exists(vertice_path):
#                     del data[key]
#                 else:
#                     if args.dataset == "vocaset":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
#                     elif args.dataset == "BIWI":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
#                     elif args.dataset == "multiface":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
#                     elif args.dataset == "multiface_2":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
#                     elif args.dataset == "3DMEAD_new":
#                         data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

#     subjects_dict = {}
#     subjects_dict["train"] = [i.split("_")[0] for i in args.train_subjects.split(" ")]  
#     subjects_dict["val"] = [i.split("_")[0] for i in args.val_subjects.split(" ")]  
#     subjects_dict["test"] = [i.split("_")[0] for i in args.test_subjects.split(" ")]  

#     splits = {
#         '3DMEAD_new': {
#             'default': {'train': list(range(1, 25)), 'val': list(range(25, 28)), 'test': list(range(28, 31))},
#             'emotion_0': {'train': list(range(1, 33)), 'val': list(range(33, 37)), 'test': list(range(37, 41))}
#         }
#     }
   
#     for k, v in data.items():
#         subject_id = k.split("_")[0]        
#         sentence_id = int(k.split("_")[1])
#         emotion_id = int(k.split(".")[0].split("_")[2])  # Extract emotion_id

#         if emotion_id == 0:
#             current_splits = splits[args.dataset]['emotion_0']
#         else:
#             current_splits = splits[args.dataset]['default']

#         if subject_id in subjects_dict["train"] and sentence_id in current_splits['train']:
#             train_data.append(v)
#         if subject_id in subjects_dict["val"] and sentence_id in current_splits['val']:
#             valid_data.append(v)
#         if subject_id in subjects_dict["test"] and sentence_id in current_splits['test']:
#             test_data.append(v)

#     print(len(train_data), len(valid_data), len(test_data))
#     return train_data, valid_data, test_data, subjects_dict

# def get_dataloaders(args):
#     dataset = {}
#     train_data, valid_data, test_data, subjects_dict = read_data(args)
#     train_data = Dataset(train_data,subjects_dict,"train")
#     dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
#     valid_data = Dataset(valid_data,subjects_dict,"val")
#     dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
#     test_data = Dataset(test_data,subjects_dict,"test")
#     dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
#     return dataset

# if __name__ == "__main__":
#     get_dataloaders()

import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
import random, math
from transformers import Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        sentence_id = int(file_name.split(".")[0].split("_")[1])
        emotion_id = int(file_name.split(".")[0].split("_")[2])
        intensity_id = int(file_name.split(".")[0].split("_")[3])
        if self.data_type == "train":
            subject = file_name.split("_")[0]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r, f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = key.split("_")[0]                
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path, f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    data[key]["vertice"] = np.load(vertice_path, allow_pickle=True)
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = data[key]["vertice"][::2, :]  # due to memory limit
                    if args.dataset == "3DMEAD_new":
                        data[key]["vertice"] = np.expand_dims(data[key]["vertice"], axis=0)  # Add a dimension to match expected shape

    subjects_dict = {}
    subjects_dict["train"] = [i.split("_")[0] for i in args.train_subjects.split(" ")]  
    subjects_dict["val"] = [i.split("_")[0] for i in args.val_subjects.split(" ")]  
    subjects_dict["test"] = [i.split("_")[0] for i in args.test_subjects.split(" ")]  

    splits = {
        '3DMEAD_new': {
            'default': {'train': list(range(1, 25)), 'val': list(range(25, 28)), 'test': list(range(28, 31))},
            'emotion_0': {'train': list(range(1, 33)), 'val': list(range(33, 37)), 'test': list(range(37, 41))}
        }
    }
   
    for k, v in data.items():
        subject_id = k.split("_")[0]        
        sentence_id = int(k.split("_")[1])
        emotion_id = int(k.split(".")[0].split("_")[2])  # Extract emotion_id

        if emotion_id == 0:
            current_splits = splits[args.dataset]['emotion_0']
        else:
            current_splits = splits[args.dataset]['default']

        if subject_id in subjects_dict["train"] and sentence_id in current_splits['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in current_splits['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in current_splits['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
