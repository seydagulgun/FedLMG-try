import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import os
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as tvu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import copy
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import partition,Truncated,evaluation
from datasets.DomainNet import get_all_domainnet_dloader
import torchvision

OUTPUT_DIR = 'output'
BACKBONES = ['mobilenetv3', 'resnet18', 'resnet34', 'mobilenetv2', 'vgg16', 'shufflenet']

def _load_teacher(num_classes, client_idx):
    backbone = BACKBONES[client_idx]
    path = f'{OUTPUT_DIR}/nicopp_img10_{client_idx}_epoch_19_{backbone}.tar'
    model = ServerTune(classes=num_classes)
    if backbone == 'mobilenetv3':
        model.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.encoder.classifier[3] = torch.nn.Identity()
        model.final_proj = nn.Linear(1024, num_classes)
    elif backbone == 'resnet18':
        model.encoder = torchvision.models.resnet18(pretrained=True)
        model.encoder.fc = torch.nn.Identity()
        model.final_proj = nn.Linear(512, num_classes)
    elif backbone == 'resnet34':
        model.encoder = torchvision.models.resnet34(pretrained=True)
        model.encoder.fc = torch.nn.Identity()
        model.final_proj = nn.Linear(512, num_classes)
    elif backbone == 'mobilenetv2':
        model.encoder = torchvision.models.mobilenet_v2(pretrained=True)
        model.encoder.classifier[1] = torch.nn.Identity()
        model.final_proj = nn.Linear(1280, num_classes)
    elif backbone == 'vgg16':
        model.encoder = torchvision.models.vgg16(pretrained=True)
        model.encoder.classifier[6] = torch.nn.Identity()
        model.final_proj = nn.Linear(4096, num_classes)
    elif backbone == 'shufflenet':
        model.encoder = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.encoder.fc = torch.nn.Identity()
        model.final_proj = nn.Linear(1024, num_classes)
    load_state_dict(model, torch.load(path, map_location='cpu'), prefix='')
    return model.cuda().eval()

class Server: # as a user
    def __init__(self,transform,bs,classes,imgpath='data'):
        self.dataset = None
        self.args = {'trans':transform,'bs':bs}
        self.dataloader = None
        self.classes = classes
        self.model = ServerTune(classes = classes).cuda()
        self.num_classes = classes
        self.imgpath = imgpath
        self.start_class = 0

        path = "/home/seyda/FedLMG-try/FedLMG/datasets/NICO_DG/autumn"
        f = os.listdir(path)
        for i in range(len(f)):
            f[i] = 'an image of ' + f[i].lower()
        self.class_prompts = sorted(f)


    def update_features(self,dataloader=None):
        if dataloader !=None:
            self.dataloader = dataloader
        else:
            self.dataset = ServerData_read(f'/home/share/gen_data_nips/{self.imgpath}',self.args['trans'])
            self.dataloader = DataLoader(self.dataset,batch_size=self.args['bs'],shuffle=True,num_workers=8,pin_memory=True,drop_last=False)

    def train(self,lr,epochs,test_data):
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)
        print(len(self.dataloader))
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for i, (image, label) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.model(image)
                loss = task_criterion(output,label)
                loss.backward()
                optimizer.step()
                if i %50 ==0:
                    print('step',i,'loss',loss)

            if epoch %5 ==0:
                top1, topk = evaluation(self.model,test_data)
                print(f'final server model: top1 {top1}, top5 {topk}')

    def multi_tea_kd_train(self,lr,epochs,test_data):
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)
        teachermodels = [_load_teacher(self.num_classes, i) for i in range(6)]

        print(len(self.dataloader))
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for i, (image, label,_) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()

                knowledge_list = []
                for model in teachermodels:
                    with torch.no_grad():
                        out = model(image)
                        knowledge_list.append(torch.softmax(out, dim=1))

                output_s = torch.log_softmax(self.model(image),dim=1)
                knowledge = sum(knowledge_list)/len(knowledge_list)
                kd_loss = torch.mean(torch.sum(-1 * knowledge * output_s, dim=1))

                ce_loss = task_criterion(output_s,label)

                loss = kd_loss+ ce_loss
                loss.backward()
                optimizer.step()
                if i %50 ==0:
                    print('step',i,'loss',loss,'kd_loss',kd_loss,'ce_loss',ce_loss)

            if epoch %5 ==0:
                top1, topk = evaluation(self.model,test_data)
                print(f'final server model: top1 {top1}, top5 {topk}')

    def sp_tea_kd_train(self,lr,epochs,test_data):
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)
        teachermodels = [_load_teacher(self.num_classes, i) for i in range(6)]

        print(len(self.dataloader))
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for i, (image, label,domain) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()
                output = self.model(image)
                output_teacher = output.detach().clone()
                for i in range(image.size()[0]):
                    with torch.no_grad():
                        teacher = teachermodels[domain[i]]
                        output_teacher[i] = teacher(image[i].unsqueeze(0))[0]

                output_s = torch.log_softmax(output,dim=1)
                output_teacher = torch.softmax(output_teacher, dim=1)
                kd_loss = torch.mean(torch.sum(-1 * output_teacher * output_s, dim=1))

                ce_loss = task_criterion(output_s,label)

                loss = kd_loss+ ce_loss
                loss.backward()
                optimizer.step()
                if i %50 ==0:
                    print('step',i,'loss',loss,'kd_loss',kd_loss,'ce_loss',ce_loss)

            if epoch %5 ==0:
                top1, topk = evaluation(self.model,test_data)
                print(f'final server model: top1 {top1}, top5 {topk}')


    def feddf(self,lr,epochs,test_data):
        task_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-5)
        teachermodels = [_load_teacher(self.num_classes, i) for i in range(6)]

        print(len(self.dataloader))
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for i, (image, label) in enumerate((self.dataloader)):
                optimizer.zero_grad()
                image = image.cuda()
                label = label.cuda()

                knowledge_list = []
                for model in teachermodels:
                    with torch.no_grad():
                        out = model(image)
                        knowledge_list.append(torch.softmax(out, dim=1))

                output_s = torch.log_softmax(self.model(image),dim=1)
                knowledge = sum(knowledge_list)/len(knowledge_list)
                kd_loss = torch.mean(torch.sum(-1 * knowledge * output_s, dim=1))
                loss = kd_loss
                loss.backward()
                optimizer.step()
                if i %50 ==0:
                    print('step',i,'loss',loss,'kd_loss',kd_loss)

            if epoch %5 ==0:
                top1, topk = evaluation(self.model,test_data)
                print(f'final server model: top1 {top1}, top5 {topk}')


    def get_client_features(self):
        return [self.client_features,]

    def aggregate(self, models):
        weights = [1/len(models)]*len(models)
        unionstate = models[0].state_dict()
        for k, client in enumerate(models):
            client_state = client.state_dict()
            for st in unionstate:
                if k==0:
                    unionstate[st] = client_state[st]*weights[k]
                else:
                    unionstate[st] += client_state[st]*weights[k]

        self.model.load_state_dict(unionstate,strict=False)

class ServerTune(nn.Module):
    def __init__(self, classes=345):
        super(ServerTune, self).__init__()

        #self.encoder = torchvision.models.resnet18(pretrained=True)
        #self.encoder.fc = torch.nn.Identity()
        #self.final_proj = nn.Sequential(nn.Linear(512,classes))

        #self.encoder = torchvision.models.resnet34(pretrained=True)
        #self.encoder.fc = torch.nn.Identity()
        #self.final_proj = nn.Sequential(nn.Linear(512,classes))

        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.encoder.fc = torch.nn.Identity()
        self.final_proj = nn.Sequential(
            nn.Linear(2048,classes)
        )

    def forward(self, x, get_fea=False,input_image=True):

        if input_image:
            with torch.no_grad():
                x =  self.encoder(x)

        if get_fea:
            return x.view(x.shape[0],-1)

        out = self.final_proj(x.view(x.shape[0],-1))

        return out

class ServerData_read(Dataset):
    def __init__(self, root_dir,transforms=None):
        super(ServerData_read, self).__init__()
        self.root_dir = root_dir

        nicopp_path = "/home/seyda/FedLMG-try/FedLMG/datasets/NICO_DG/autumn"
        f = os.listdir(nicopp_path)
        for i in range(len(f)):
            f[i] = 'an image of ' + f[i].lower()
        self.class_prompts = sorted(f)
        self.classes = {c:i for i,c in enumerate(self.class_prompts) if i<60}

        self.images = []
        self.targets = []
        self.transforms = transforms
        for c in self.classes:
            class_dir = os.path.join(self.root_dir, 'a photo of '+str(c))
            for image_name in os.listdir(class_dir):
                if '.ipynb_checkpoints' in image_name: continue
                image_path = os.path.join(class_dir, image_name)
                self.images.append(image_path)
                self.targets.append(self.classes[c])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        target = self.targets[index]
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
