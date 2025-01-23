import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from data_loader_3DMEAD import get_dataloaders
from faceformer_3DMEAD import Faceformer


# def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
#     save_path = os.path.join(args.dataset,args.save_path)
#     if os.path.exists(save_path):
#         shutil.rmtree(save_path)
#     os.makedirs(save_path)

#     train_subjects_list = [i for i in args.train_subjects.split(" ")]
#     iteration = 0
#     for e in range(epoch+1):
#         loss_log = []
#         # train
#         model.train()
#         pbar = tqdm(enumerate(train_loader),total=len(train_loader))
#         optimizer.zero_grad()

#         for i, (audio, vertice, template, one_hot, file_name) in pbar:
#             iteration += 1
#             vertice = str(vertice[0])
#             if args.dataset == 'BIWI':
#                 vertice = np.load(vertice, allow_pickle=True)
#             elif args.dataset == 'vocaset':
#                 vertice = np.load(vertice, allow_pickle=True)[::2, :]
#             elif args.dataset == 'multiface':
#                 vertice = np.load(vertice, allow_pickle=True)
#             elif args.dataset == '3DMEAD_new':
#                 if isinstance(vertice, torch.Tensor):
#                     vertice = vertice.cpu().numpy()
#                 else:
#                     vertice = np.load(vertice, allow_pickle=True)
#                 vertice = vertice.astype(np.float32)
#                 vertice = torch.from_numpy(vertice).unsqueeze(0)          
            
#             vertice = vertice.astype(np.float32)
#             vertice = torch.from_numpy(vertice)
#             vertice = torch.unsqueeze(vertice, 0)
#             # to gpu
#             audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
#             loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
#             loss.backward()
#             loss_log.append(loss.item())
#             if i % args.gradient_accumulation_steps==0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#             pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
#         # validation
#         valid_loss_log = []
#         model.eval()
#         for audio, vertice, template, one_hot_all,file_name in dev_loader:
#             vertice = str(vertice[0])
#             if args.dataset == 'BIWI':
#                 vertice = np.load(vertice, allow_pickle=True)
#             elif args.dataset == 'vocaset':
#                 vertice = np.load(vertice, allow_pickle=True)[::2, :]
#             elif args.dataset == 'multiface':
#                 vertice = np.load(vertice, allow_pickle=True)
#             elif args.dataset == '3DMEAD_new':
#                 if isinstance(vertice, torch.Tensor):
#                     vertice = vertice.cpu().numpy()
#                 else:
#                     vertice = np.load(vertice, allow_pickle=True)
#                 vertice = vertice.astype(np.float32)
#                 vertice = torch.from_numpy(vertice).unsqueeze(0)
            
#             vertice = vertice.astype(np.float32)
#             vertice = torch.from_numpy(vertice)
#             vertice = torch.unsqueeze(vertice, 0)
#             # to gpu
#             audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
#             train_subject = "_".join(file_name[0].split("_")[:-1])
#             if train_subject in train_subjects_list:
#                 condition_subject = train_subject
#                 iter = train_subjects_list.index(condition_subject)
#                 one_hot = one_hot_all[:,iter,:]
#                 loss = model(audio, template,  vertice, one_hot, criterion)
#                 valid_loss_log.append(loss.item())
#             else:
#                 for iter in range(one_hot_all.shape[-1]):
#                     condition_subject = train_subjects_list[iter]
#                     one_hot = one_hot_all[:,iter,:]
#                     loss = model(audio, template,  vertice, one_hot, criterion)
#                     valid_loss_log.append(loss.item())
                        
#         current_loss = np.mean(valid_loss_log)
        
#         if (e > 0 and e % 25 == 0) or e == args.max_epoch:
#             torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

#         print("epoch: {}, current loss:{:.7f}".format(e+1,current_loss))    
#     return model
#correct trainer
def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    save_path = os.path.join(args.dataset, args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            if args.dataset == 'BIWI':
                vertice = np.load(vertice[0], allow_pickle=True)
            elif args.dataset == 'vocaset':
                vertice = np.load(vertice[0], allow_pickle=True)[::2, :]
            elif args.dataset == 'multiface':
                vertice = np.load(vertice[0], allow_pickle=True)
            elif args.dataset == '3DMEAD_new':
                if isinstance(vertice, torch.Tensor):
                    vertice = vertice.cpu().numpy()
                else:
                    vertice = np.load(vertice[0], allow_pickle=True)
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice).unsqueeze(0)

            vertice = vertice.float()  # Convert tensor to float
            vertice = torch.unsqueeze(vertice, 0)
            # to gpu
            audio, vertice, template, one_hot = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration, np.mean(loss_log)))
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            if args.dataset == 'BIWI':
                vertice = np.load(vertice[0], allow_pickle=True)
            elif args.dataset == 'vocaset':
                vertice = np.load(vertice[0], allow_pickle=True)[::2, :]
            elif args.dataset == 'multiface':
                vertice = np.load(vertice[0], allow_pickle=True)
            elif args.dataset == '3DMEAD_new':
                if isinstance(vertice, torch.Tensor):
                    vertice = vertice.cpu().numpy()
                else:
                    vertice = np.load(vertice[0], allow_pickle=True)
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice).unsqueeze(0)

            vertice = vertice.float()  # Convert tensor to float
            vertice = torch.unsqueeze(vertice, 0)
            # to gpu
            audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]
                loss = model(audio, template, vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:, iter, :]
                    loss = model(audio, template, vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epoch: {}, current loss:{:.7f}".format(e+1, current_loss))
    return model


# def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
#     save_path = os.path.join(args.dataset, args.save_path)
#     if os.path.exists(save_path):
#         shutil.rmtree(save_path)
#     os.makedirs(save_path)

#     train_subjects_list = [i for i in args.train_subjects.split(" ")]
#     iteration = 0
#     for e in range(epoch + 1):
#         loss_log = []
#         # train
#         model.train()
#         pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#         optimizer.zero_grad()

#         for i, (audio, vertice, template, one_hot, file_name) in pbar:
#             iteration += 1
#             if isinstance(vertice[0], torch.Tensor):
#                 vertice = vertice[0].cpu().numpy()
#             else:
#                 vertice = np.load(vertice[0], allow_pickle=True)
#             vertice = vertice.astype(np.float32)
#             vertice = torch.from_numpy(vertice).unsqueeze(0)

#             # to gpu
#             audio, vertice, template, one_hot = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
#             loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
#             loss.backward()
#             loss_log.append(loss.item())
#             if i % args.gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#             pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e + 1), iteration, np.mean(loss_log)))

#         # validation
#         valid_loss_log = []
#         model.eval()
#         for audio, vertice, template, one_hot_all, file_name in dev_loader:
#             if isinstance(vertice[0], torch.Tensor):
#                 vertice = vertice[0].cpu().numpy()
#             else:
#                 vertice = np.load(vertice[0], allow_pickle=True)
#             vertice = vertice.astype(np.float32)
#             vertice = torch.from_numpy(vertice).unsqueeze(0)

#             # to gpu
#             audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
#             train_subject = "_".join(file_name[0].split("_")[:-1])
#             if train_subject in train_subjects_list:
#                 condition_subject = train_subject
#                 iter = train_subjects_list.index(condition_subject)
#                 one_hot = one_hot_all[:, iter, :]
#                 loss = model(audio, template, vertice, one_hot, criterion)
#                 valid_loss_log.append(loss.item())
#             else:
#                 for iter in range(one_hot_all.shape[-1]):
#                     condition_subject = train_subjects_list[iter]
#                     one_hot = one_hot_all[:, iter, :]
#                     loss = model(audio, template, vertice, one_hot, criterion)
#                     valid_loss_log.append(loss.item())

#         current_loss = np.mean(valid_loss_log)

#         if (e > 0 and e % 25 == 0) or e == args.max_epoch:
#             torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

#         print("epoch: {}, current loss:{:.7f}".format(e + 1, current_loss))
#     return model

@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        if args.dataset == 'BIWI':
            vertice = np.load(str(vertice[0]), allow_pickle=True)
        elif args.dataset == 'vocaset':
            vertice = np.load(str(vertice[0]), allow_pickle=True)[::2, :]
        elif args.dataset == 'multiface':
            vertice = np.load(str(vertice[0]), allow_pickle=True)
        elif args.dataset == '3DMEAD_new':
            if isinstance(vertice, torch.Tensor):
                vertice = vertice.cpu().numpy()
            else:
                vertice = np.load(str(vertice[0]), allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice).unsqueeze(0)

        vertice = vertice.float()  # Convert tensor to float
        vertice = torch.unsqueeze(vertice, 0)
        # to gpu
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())

# @torch.no_grad()
# def test(args, model, test_loader, epoch):
#     result_path = os.path.join(args.dataset, args.result_path)
#     if os.path.exists(result_path):
#         shutil.rmtree(result_path)
#     os.makedirs(result_path)

#     save_path = os.path.join(args.dataset, args.save_path)
#     train_subjects_list = [i for i in args.train_subjects.split(" ")]

#     model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
#     model = model.to(torch.device("cuda"))
#     model.eval()

#     for audio, vertice, template, one_hot_all, file_name in test_loader:
#         if isinstance(vertice[0], torch.Tensor):
#             vertice = vertice[0].cpu().numpy()
#         else:
#             vertice = np.load(vertice[0], allow_pickle=True)
#         vertice = vertice.astype(np.float32)
#         vertice = torch.from_numpy(vertice).unsqueeze(0)

#         # to gpu
#         audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
#         train_subject = "_".join(file_name[0].split("_")[:-1])
#         if train_subject in train_subjects_list:
#             condition_subject = train_subject
#             iter = train_subjects_list.index(condition_subject)
#             one_hot = one_hot_all[:, iter, :]
#             prediction = model.predict(audio, template, one_hot)
#             prediction = prediction.squeeze()  # (seq_len, V*3)
#             np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())
#         else:
#             for iter in range(one_hot_all.shape[-1]):
#                 condition_subject = train_subjects_list[iter]
#                 one_hot = one_hot_all[:, iter, :]
#                 prediction = model.predict(audio, template, one_hot)
#                 prediction = prediction.squeeze()  # (seq_len, V*3)
#                 np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="3DMEAD_new", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="processed_new", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates_3DMEAD_new.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save_3DMEAD", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result_3DMEAD", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")
    parser.add_argument("--val_subjects", type=str,  default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")
    parser.add_argument("--test_subjects", type=str, default= "M003 M005 M007 M009 M011 M012 M013 M019 M022 M023 M024 M025 M026 M027 M028 M029 M030 M031 M032 M033 M034 M035 M037 M039 M040 M041 M042 W009 W011 W014 W015 W016 W018 W019 W021 W023 W024 W025 W026 W028 W029 W033 W035 W036 W037 W038 W040")
    parser.add_argument("--load_epoch", type=int, required=True, help='epoch of the model to load for testing')
    args = parser.parse_args()
    
    # Build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # To cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    # Load data
    dataset = get_dataloaders(args)

    # Load pre-trained model
    save_path = os.path.join(args.dataset, args.save_path)
    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(args.load_epoch))))

    # Test the model
    test(args, model, dataset["test"], epoch=args.load_epoch)
    
if __name__ == "__main__":
    main()
