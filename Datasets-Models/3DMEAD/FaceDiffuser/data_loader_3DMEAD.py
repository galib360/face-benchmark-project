import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
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

        if self.data_type == "train":
            subject = file_name.split("_")[0]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels

        return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(
            one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_path, args.dataset, args.wav_path)
    vertices_path = os.path.join(args.data_path, args.dataset, args.vertices_path)

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/hubert-xlarge-ls960-ft")  # HuBERT uses the processor of Wav2Vec 2.0

    template_file = os.path.join(args.data_path, args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    indices_to_split = []
    all_subjects = args.test_subjects.split() + args.val_subjects.split() + args.test_subjects.split()
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = key.split("_")[0]                
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                    # print("Vertices Data Not Found! ", vertice_path)
                else:
                    data[key]["vertice"] = vertice_path

    indices_to_split = np.array(indices_to_split)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
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



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker,
                                       generator=g)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()

