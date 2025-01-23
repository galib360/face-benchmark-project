import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import shutil

import pandas as pd
import torch
import numpy as np

from data_loader_3DMEAD import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from utils import *


@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)

    model.load_state_dict(torch.load(os.path.join(save_path, f'{args.model}_{args.dataset}_{epoch}.pth')))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000

    # Define the subjects for which you want to generate samples
    target_subjects = ["M003", "M005", "M007", "M009", "M011", "M012", "M013", "M019", "M022", "M023", "M024", "M025", "M026", "M027", "M028", "M029", "M030", "M031", "M032", "M033", "M034", "M035", "M037", "M039", "M040", "M041", "M042", "W009", "W011", "W014", "W015", "W016", "W018", "W019", "W021", "W023", "W024", "W025", "W026", "W028", "W029", "W033", "W035", "W036", "W037", "W038", "W040"]

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        train_subject = file_name[0].split("_")[0]

        # Skip subjects that are not in the target_subjects list
        if train_subject not in target_subjects:
            continue

        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        if args.dataset == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)

        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        vertice_path = os.path.split(vertice_path)[-1][:-4]
        print(vertice_path)

        condition_subject = train_subject
        iter = target_subjects.index(condition_subject)
        one_hot = one_hot_all[:, iter, :]
        one_hot = one_hot.to(device=device)

        for sample_idx in range(1, args.num_samples + 1):
            sample = diffusion.p_sample_loop(
                model,
                shape,
                clip_denoised=False,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                },
                skip_timesteps=args.skip_steps,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                device=device
            )
            sample = sample.squeeze()
            sample = sample.detach().cpu().numpy()

            out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
            if 'damm' in args.dataset:
                sample = RIG_SCALER.inverse_transform(sample)
                np.save(os.path.join(args.result_path, out_path), sample)
                df = pd.DataFrame(sample)
                df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
            else:
                np.save(os.path.join(args.result_path, out_path), sample)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="3DMEAD_new", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=15069, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=512, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=512, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="processed_new", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates_3DMEAD_new.pkl", help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save_3DMEAD_4", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="3DMEAD_samples_4", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")
    parser.add_argument("--val_subjects", type=str,  default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")
    parser.add_argument("--test_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")    
    # parser.add_argument("--train_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 W009 W011 W014 W015 W016 W018 W019")
    # parser.add_argument("--val_subjects", type=str,  default= "M003 M005 M007 M009 M011 M012 M013 W009 W011 W014 W015 W016 W018 W019")
    # parser.add_argument("--test_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 W009 W011 W014 W015 W016 W018 W019")
    # parser.add_argument("--train_subjects", type=str, default= "M032 M035 W033 W035")
    # parser.add_argument("--val_subjects", type=str,  default= "M032 M035 W033 W035")
    # parser.add_argument("--test_subjects", type=str, default= "M032 M035 W033 W035")
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=10, help='number of samples to generate per audio')
    args = parser.parse_args()

    # # assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    # if 'damm' in args.dataset:
    #     model = FaceDiffDamm(args)
    # elif 'beat' in args.dataset:
    #     model = FaceDiffBeat(
    #             args,
    #             vertice_dim=args.vertice_dim,
    #             latent_dim=args.feature_dim,
    #             diffusion_steps=args.diff_steps,
    #             gru_latent_dim=args.gru_dim,
    #             num_layers=args.gru_layers,
    #         )
    # else:
    #     model = FaceDiff(
    #         args,
    #         vertice_dim=args.vertice_dim,
    #         latent_dim=args.feature_dim,
    #         diffusion_steps=args.diff_steps,
    #         gru_latent_dim=args.gru_dim,
    #         num_layers=args.gru_layers,
    #     )
    # print("model parameters: ", count_parameters(model))
    # cuda = torch.device(args.device)

    # model = model.to(cuda)
    # dataset = get_dataloaders(args)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # model = trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer,
    #                      epoch=args.max_epoch, device=args.device)
    # test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)
     # Load pre-trained model
    model_path = "/storage/scratch/2025930/Thesis/Models/FaceDiffuser/save_3DMEAD_4/face_diffuser_3DMEAD_new_50.pth"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    state_dict = torch.load(model_path)

    # Initialize the model instance
    if 'damm' in args.dataset:
        model = FaceDiffDamm(args)
    elif 'beat' in args.dataset:
        model = FaceDiffBeat(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )
    else:
        model = FaceDiff(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )

    # Load state dictionary into the model
    model.load_state_dict(state_dict)

    # Set device
    device = torch.device(args.device)
    model = model.to(device)

    # Get dataloaders
    dataset = get_dataloaders(args)

    # Skip the optimizer and training function calls
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # model = trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer, epoch=args.max_epoch, device=args.device)

    # Call testing function
    test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)




if __name__ == "__main__":
    main()