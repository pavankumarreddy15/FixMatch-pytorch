import argparse
from ast import arg, parse
import logging
import math
import os
import random
import shutil
from statistics import mode
import time
from collections import OrderedDict
import json
from dataset.randaugment import RandAugmentMC
from PIL import Image


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from train import train

from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel
from submodlib.functions.disparityMin import DisparityMinFunction
from submodlib.functions.logDeterminant import LogDeterminantFunction
from submodlib.functions.graphCut import GraphCutFunction
from submodlib.functions.facilityLocationConditionalMutualInformation import FacilityLocationConditionalMutualInformationFunction 
from submodlib.functions.logDeterminantConditionalMutualInformation import LogDeterminantConditionalMutualInformationFunction
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from submodlib.functions.facilityLocationVariantMutualInformation import FacilityLocationVariantMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import LogDeterminantMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import GraphCutMutualInformationFunction

from distil.distil.active_learning_strategies.badge import BADGE
from distil.distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.distil.active_learning_strategies.strategy import Strategy

from trust.trust.utils.custom_dataset import load_dataset_custom

from models.wideresnet import build_wideresnet

from models.ema import ModelEMA

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
parser.add_argument('--alg', default='fixmatch', type=str,help='Algorithm to be used from [fixmatch,supervised]')
parser.add_argument('--setting', default='random', type=str,help='setting is subset selection MI method')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000,
                    help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--arch', default='wideresnet', type=str,
                    choices=['wideresnet', 'resnext'],
                    help='dataset name')
parser.add_argument('--total-steps', default=2**20, type=int,
                    help='number of total steps to run')
parser.add_argument('--supervised-steps',default=2**12,type=int,help='number of supervised steps to run')
parser.add_argument('--supervised-evalstep',default=2**7,type=int,help='number of supervised eval steps to run')
parser.add_argument('--eval-step', default=1024, type=int,
                    help='number of eval steps to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    help='initial learning rate')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--use-ema', action='store_true', default=True,
                    help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--out', default='result',
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help="random seed")
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")
parser.add_argument('--rounds', default=10, type=int,help='number of active learning rounds')
parser.add_argument('--imbalance', default=10, type=int,help='imbalance ratio')
parser.add_argument('--run',default=1,type=int,help='number of run performing')
parser.add_argument('--num_classes',default=10,type=int,help='number of classes in dataset')
parser.add_argument('--model_width',default=2,type=int,help="width of wrn used")
parser.add_argument('--model_depth',default=28,type=int,help="depth of wrn used")
parser.add_argument('--gpudevice',default=0,type=int,help="gpu device to be used")

args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self,transform=None,target_transform=None):
        super().init(self,transform=None,target_transform=None)
    def __getitem__(self,index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,target
      
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None, labeled=True):
        self.labeled = labeled
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        
        if self.transform:
            x = Image.fromarray(np.array(self.data[index]).astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        return x,y
    
    def __len__(self):
        return len(self.data)
      
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def counter(labeled_indices,labels):
    d = {}
    for x in labeled_indices:
        if labels[x] in d:
            d[labels[x]] += 1
        else:
            d[labels[x]] = 1
    # print(d)
    return d 

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def _load_cifar10():
    splits = {}
    x = 4500
    y = x//args.imbalance
    rare_classes = list(range(5))
    per_class_train = [x for i in range(10)]
    for i in rare_classes:
        per_class_train[i]=y
    val_x = 500
    val_y = val_x//args.imbalance
    per_class_val = [val_x for i in range(10)]
    for i in rare_classes:
        per_class_val[i]=val_y
    per_class_lake  = [0 for i in range(10)]
    print(per_class_train)
    sel_cls_idx = list(range(5))
    split_cfg = {"per_imbclass_train":450,"per_imbclass_val":50,"per_imbclass_lake":0,"per_class_train":per_class_train,"per_class_val":per_class_val,"per_class_lake":per_class_lake,"sel_cls_idx":[3,5,8]}
    train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls = load_dataset_custom("data","cifar10","classimb",split_cfg,augVal=False,dataAug=False)
    train_data = {}
    test_data  = {}
    val_data   = {}
    train_data["images"] = np.array([np.array(train_set[i][0]) for i in range(len(train_set))])
    train_data["labels"] = np.array([int(train_set[i][1]) for i in range(len(train_set))])
    test_data["images"]  = np.array([np.array(test_set[i][0]) for i in range(len(test_set))])
    test_data["labels"]  = np.array([int(test_set[i][1]) for i in range(len(test_set))])
    val_data["images"]   = np.array([np.array(val_set[i][0]) for i in range(len(val_set))])
    val_data["labels"]   = np.array([int(val_set[i][1]) for i in range(len(val_set))])

    splits["train"] = train_data
    splits["test"]  = test_data
    splits["val"]   = val_data
    return train_data,test_data,val_data,sel_cls_idx

def _load_svhn():
    splits = {}
    per_class_train = [730,730,730,730,730,7300,7300,7300,7300,7300]
    sel_cls_idx = list(range(5))
    per_class_val   = [700 for i in range(10)]
    per_class_lake  = [0 for i in range(10)]
    split_cfg = {"per_imbclass_train":2500,"per_imbclass_val":500,"per_imbclass_lake":0,"per_class_train":per_class_train,"per_class_val":per_class_val,"per_class_lake":per_class_lake,"sel_cls_idx":[3,5,8]}
    train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls = load_dataset_custom("data","svhn","classimb",split_cfg,augVal=False,dataAug=False)
    train_data = {}
    test_data  = {}
    val_data   = {}
    train_data["images"] = np.array([np.array(train_set[i][0]) for i in range(len(train_set))])
    train_data["labels"] = np.array([int(train_set[i][1]) for i in range(len(train_set))])
    test_data["images"]  = np.array([np.array(test_set[i][0]) for i in range(len(test_set))])
    test_data["labels"]  = np.array([int(test_set[i][1]) for i in range(len(test_set))])
    val_data["images"]   = np.array([np.array(val_set[i][0]) for i in range(len(val_set))])
    val_data["labels"]   = np.array([int(val_set[i][1]) for i in range(len(val_set))])

    splits["train"] = train_data
    splits["test"]  = test_data
    splits["val"]   = val_data
    return train_data,test_data,sel_cls_idx

def select_subset(labeled_indices,unlabeled_indices,features,features_available,num_labels,numclasses,labels,images,total_size,setting,strat,model):
    if features_available and setting=="FL2MI":
        classes = np.unique(labels)
        rem_index = unlabeled_indices.copy()
        
        partial_labeled_indices = labeled_indices.copy()
        l_labels = list(labels[labeled_indices])
        #unlabeled_features = features[rem_index]
        fulldata = features
        K_dense2 = create_kernel(fulldata,mode='dense',metric='cosine') #similarity matrix for whole data used to extract similarity matrices for query and private datas
        K_dense = K_dense2[rem_index][:,rem_index]
        print("data_sijs created using create_kernel")
        for c in classes:
            cls_labeled_indices = [partial_labeled_indices[i] for i in range(len(partial_labeled_indices)) if l_labels[i]==c]
            budget = num_labels//(numclasses*args.rounds)
            num_queries = len(cls_labeled_indices)
            #num_privates = len(private_indices)
            query_sijs = K_dense2[rem_index][:,cls_labeled_indices]
            data_size = len(rem_index)
            #private_sijs = K_dense2[rem_index][:,private_indices]
            print(K_dense.shape,query_sijs.shape)
            obj1 = FacilityLocationVariantMutualInformationFunction(n=data_size,num_queries=num_queries,query_sijs=query_sijs,data=None,queryData=None,metric='cosine',queryDiversityEta=10)
            print("obj"+str(c)+"instantiated")
            greedyList = obj1.maximize(budget=budget,optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
            print("greedylist obtained")
            train_idx = [rem_index[i[0]] for i in greedyList]
            partial_labeled_indices += list(train_idx)
            rem_index = list(set(unlabeled_indices)-set(partial_labeled_indices))
            #rem_index = unlabeled_indices.copy()
            K_dense = K_dense2[rem_index][:,rem_index]
            l_labels += list(labels[train_idx])
        labeled_indices = partial_labeled_indices
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    elif features_available and setting=="FL1MI":
        classes = np.unique(labels)
        rem_index = unlabeled_indices.copy()
        
        partial_labeled_indices = labeled_indices.copy()
        l_labels = list(labels[labeled_indices])
        #unlabeled_features = features[rem_index]
        fulldata = features
        K_dense2 = create_kernel(fulldata,mode='dense',metric='cosine') #similarity matrix for whole data used to extract similarity matrices for query and private datas
        K_dense = K_dense2[rem_index][:,rem_index]
        print("data_sijs created using create_kernel")
        for c in classes:
            cls_labeled_indices = [partial_labeled_indices[i] for i in range(len(partial_labeled_indices)) if l_labels[i]==c]
            budget = num_labels//(numclasses*args.rounds)
            num_queries = len(cls_labeled_indices)
            #num_privates = len(private_indices)
            query_sijs = K_dense2[rem_index][:,cls_labeled_indices]
            data_sijs = K_dense2[rem_index][:,rem_index]
            data_size = len(rem_index)
            #private_sijs = K_dense2[rem_index][:,private_indices]
            print(K_dense.shape,query_sijs.shape)
            obj1 = FacilityLocationMutualInformationFunction(n=data_size,num_queries=num_queries,data_sijs=data_sijs,query_sijs=query_sijs,data=None,queryData=None,metric='cosine',magnificationEta=1)
            print("obj"+str(c)+"instantiated")
            greedyList = obj1.maximize(budget=budget,optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
            print("greedylist obtained")
            train_idx = [rem_index[i[0]] for i in greedyList]
            partial_labeled_indices += list(train_idx)
            rem_index = list(set(unlabeled_indices)-set(partial_labeled_indices))
            #rem_index = unlabeled_indices.copy()
            K_dense = K_dense2[rem_index][:,rem_index]
            l_labels += list(labels[train_idx])
        labeled_indices = partial_labeled_indices
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    elif features_available and setting=="GCMI":
        classes = np.unique(labels)
        rem_index = unlabeled_indices.copy()
        
        partial_labeled_indices = labeled_indices.copy()
        l_labels = list(labels[labeled_indices])
        #unlabeled_features = features[rem_index]
        fulldata = features
        K_dense2 = create_kernel(fulldata,mode='dense',metric='cosine') #similarity matrix for whole data used to extract similarity matrices for query and private datas
        K_dense = K_dense2[rem_index][:,rem_index]
        print("data_sijs created using create_kernel")
        for c in classes:
            cls_labeled_indices = [partial_labeled_indices[i] for i in range(len(partial_labeled_indices)) if l_labels[i]==c]
            budget = num_labels//(numclasses*args.rounds)
            num_queries = len(cls_labeled_indices)
            #num_privates = len(private_indices)
            query_sijs = K_dense2[rem_index][:,cls_labeled_indices]
            data_size = len(rem_index)
            #private_sijs = K_dense2[rem_index][:,private_indices]
            print(K_dense.shape,query_sijs.shape)
            obj1 = GraphCutMutualInformationFunction(n=data_size,num_queries=num_queries,query_sijs=query_sijs,data=None,queryData=None,metric='cosine')
            print("obj"+str(c)+"instantiated")
            greedyList = obj1.maximize(budget=budget,optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
            print("greedylist obtained")
            train_idx = [rem_index[i[0]] for i in greedyList]
            partial_labeled_indices += list(train_idx)
            rem_index = list(set(unlabeled_indices)-set(partial_labeled_indices))
            #rem_index = unlabeled_indices.copy()
            K_dense = K_dense2[rem_index][:,rem_index]
            l_labels += list(labels[train_idx])
        labeled_indices = partial_labeled_indices
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    elif features_available and setting=="logdetMI":
        classes = np.unique(labels)
        rem_index = unlabeled_indices.copy()
        
        partial_labeled_indices = labeled_indices.copy()
        l_labels = list(labels[labeled_indices])
        #unlabeled_features = features[rem_index]
        fulldata = features
        K_dense2 = create_kernel(fulldata,mode='dense',metric='cosine') #similarity matrix for whole data used to extract similarity matrices for query and private datas
        K_dense = K_dense2[rem_index][:,rem_index]
        print("data_sijs created using create_kernel")
        for c in classes:
            cls_labeled_indices = [partial_labeled_indices[i] for i in range(len(partial_labeled_indices)) if l_labels[i]==c]
            budget = num_labels//(numclasses*args.rounds)
            num_queries = len(cls_labeled_indices)
            #num_privates = len(private_indices)
            query_sijs = K_dense2[rem_index][:,cls_labeled_indices]
            query_query_sijs = K_dense2[cls_labeled_indices][:,cls_labeled_indices]
            data_sijs = K_dense2[rem_index][:,rem_index]
            data_size = len(rem_index)
            #private_sijs = K_dense2[rem_index][:,private_indices]
            print(K_dense.shape,query_sijs.shape)
            obj1 = LogDeterminantMutualInformationFunction(n=data_size,num_queries=num_queries,lambdaVal=1,data_sijs=data_sijs,query_sijs=query_sijs,query_query_sijs=query_query_sijs,data=None,queryData=None,metric='cosine')
            print("obj"+str(c)+"instantiated")
            greedyList = obj1.maximize(budget=budget,optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
            print("greedylist obtained")
            train_idx = [rem_index[i[0]] for i in greedyList]
            partial_labeled_indices += list(train_idx)
            rem_index = list(set(unlabeled_indices)-set(partial_labeled_indices))
            #rem_index = unlabeled_indices.copy()
            K_dense = K_dense2[rem_index][:,rem_index]
            l_labels += list(labels[train_idx])
        labeled_indices = partial_labeled_indices
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    elif setting=="BADGE" and features_available:
        strat.update_model(model)
        budget = num_labels//args.rounds
        idx = strat.select(budget)
        rem_idx = unlabeled_indices.copy()
        train_idx = [rem_idx[i] for i in idx]
        labeled_indices += train_idx
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    elif setting=="US" and features_available:
        print("US")
        strat.update_model(model)
        budget = num_labels//args.rounds
        idx = strat.select(budget)
        rem_idx = unlabeled_indices.copy()
        train_idx = [rem_idx[i] for i in idx]
        labeled_indices += train_idx
        unlabeled_indices = list(set(range(total_size))-set(labeled_indices))
        return labeled_indices,unlabeled_indices
    else:
        n_labels_per_iter = num_labels//args.rounds
        np.random.seed(args.seed)
        train_idx = list(np.random.choice(np.array(unlabeled_indices), size=n_labels_per_iter, replace=False))
        labeled_indices += train_idx
        unlabeled_indices = list(set(unlabeled_indices)-set(labeled_indices))
        #when calling function call as select_subset([],fullindices,features can be None,features_available=False,num_labels,datasize,numclasses,labels,images)
        # classes = np.arange(numclasses)
        # n_labels_per_cls = num_labels//(numclasses*args.rounds)
        # for c in classes:
        #     c_indices = list(np.where(labels == c)[0])
        #     labeled_indices += c_indices[:n_labels_per_cls]
        # unlabeled_indices = list(set(unlabeled_indices)-set(labeled_indices))
        return labeled_indices,unlabeled_indices


if __name__ == "__main__":
    if args.dataset == "cifar10":
        train_set, test_set, validation_set,sel_cls_idx = _load_cifar10()
    elif args.dataset == "svhn":
        train_set, test_set, sel_cls_idx = _load_svhn()
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(train_set["images"]))
    train_set["images"] = train_set["images"][indices]
    train_set["labels"] = train_set["labels"][indices]
    

    _DATA_DIR = "data"
    _EXP_DATA_DIR = os.path.join(_DATA_DIR, args.dataset, str(args.num_labeled), args.setting)
    if(not(os.path.exists(_EXP_DATA_DIR))):
        os.makedirs(_EXP_DATA_DIR)

    if not os.path.exists(os.path.join(_EXP_DATA_DIR, args.dataset)):
        os.mkdir(os.path.join(_EXP_DATA_DIR, args.dataset))
    np.save(os.path.join(_EXP_DATA_DIR, "train_"+args.setting), train_set)
    np.save(os.path.join(_EXP_DATA_DIR, "val_"+args.setting), validation_set)
    np.save(os.path.join(_DATA_DIR, args.dataset, "test"), test_set)
    
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    normal_mean = (0.5, 0.5, 0.5)
    normal_std = (0.5, 0.5, 0.5)
    
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    #Logs 
    logs = {}
    logs["dataset"] = args.dataset
    logs["feature"] = "classimb"
    logs["sel_func"] = args.setting
    logs["sel_budget"] = args.num_labeled//args.rounds
    logs["num_selections"] = args.rounds
    logs["model"] = "wrn"
    logs["sel_cls_idx"] = sel_cls_idx

    setting = {}
    setting["imbalance_ratio"] = args.imbalance
    setting["rare_classes"] = logs["sel_cls_idx"]
    logs["setting"] = setting

    algo = args.alg

    print("------------------------Files are downloaded and saved---------------------------------")

    unlabeled_indices = np.arange(len(train_set))
    labeled_indices   = []
    n_labels = args.num_labeled
    
    np.save(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting), labeled_indices)
    np.save(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting), unlabeled_indices)
    
    print("-------------------------------Index files stored---------------------------------")

    #setting device
    if torch.cuda.is_available():
        device = "cuda:"+str(args.gpudevice)
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    
    args.device = device

    counter_map = {}
    exp_name = ""
    exp_name += str(args.dataset) + "_"
    exp_name += "labeled" + str(n_labels) + "_" + str(args.setting) + "_"
    exp_name += str(args.alg) + "_"
    exp_name += "run"+str(args.run)
    exp_dir = os.path.join(args.out, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    numrounds = args.rounds

    cur_round = 1
    tot_steps = args.total_steps
    eval_step = args.eval_step

    images = train_set["images"]
    labels = train_set["labels"]
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    l_features = []
    u_features = []
    l_indices = []
    u_indices = []
    true_u_labels = []
    classes = np.unique(labels)
    numclasses = classes.shape[0]
    datasize = labels.shape[0]

    print("------------------------Active learning SSL started---------------------------------")

    final_val_acc = []
    test_accs = []
    all_cls_acc = [[] for i in range(numclasses)]
    sel_per_cls = []
    train_dataset = [train_set['images'][i].astype(np.float32) for i in range(len(train_set['images']))]
    test_dataset = [[test_set['images'][i].astype(np.float32),test_set['labels'][i]] for i in range(len(test_set['images']))]
    test_x = torch.Tensor([np.array(test_set['images'][i].astype(np.float32)) for i in range(len(test_set['images']))])
    test_y = torch.Tensor([np.array([test_set['labels'][i]]).astype(np.int64) for i in range(len(test_set['images']))]).to(torch.long)
    my_testdataset = MyDataset(data=test_x,targets=test_y, transform=transform_val)
#     my_testdataset = CustomDataset(my_testdataset,transform=TransformFixMatch)
    test_loader = DataLoader(my_testdataset,sampler=SequentialSampler(test_dataset),batch_size=args.batch_size,num_workers=args.num_workers)
    labeled_dataset = [[train_set['images'][i].astype(np.float32),train_set['labels'][i]] for i in range(len(train_set['images']))]
    # net = wrn.WRN(2, dataset_cfg["num_classes"], args.input_channels)
    net = build_wideresnet(depth=args.model_depth,widen_factor=args.model_width,dropout=0,num_classes=args.num_classes)
    net.to(device)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)
    ema_model = ModelEMA(args, net, args.ema_decay)

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            net, optimizer, opt_level=args.opt_level)
    if args.local_rank != -1:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
    
    strat = None


    if args.setting=="BADGE":
        strategy_args = {'batch_size': 20,'device': device}
        strat = BADGE(labeled_dataset=[],unlabeled_dataset=train_dataset,net=net,nclasses=numclasses,args=strategy_args)
    elif args.setting=="US":
        strategy_args = {'batch_size': 20,'device': device}
        strat = EntropySampling(labeled_dataset=[],unlabeled_dataset=train_dataset,net=net,nclasses=numclasses,args=strategy_args)

    while cur_round <= numrounds:
        temp_exp_name = ""+str(args.dataset)+"_"+"labeled" + str(cur_round*(args.num_labeled)//(args.rounds))+ "_" + str(args.setting) + "_" + "supervised" + "_"
        temp_exp_name += "run"+str(args.run)
        temp_exp_name += "_round"+str(cur_round)
        s_temp_exp_name = ""+str(args.dataset)+"_"+"labeled" + str(cur_round*(args.num_labeled)//(args.rounds))+ "_" + str(args.setting) + "_" + "supervised" + "_"
        s_temp_exp_name += "run"+str(args.run)
        s_temp_exp_name += "_round"+str(cur_round)
        prev_counter_map = counter(labeled_indices=labeled_indices,labels=labels)
        prev_counter = []
        for i in range(numclasses):
            if i in prev_counter_map:
                prev_counter.append(prev_counter_map[i])
            else:
                prev_counter.append(0)
        if cur_round == 1:
            #Select B/numrounds number of points uniformly from all classes 
            print("------------------------Subset selection in process in "+ str(cur_round)+"th round---------------------------------")
            labeled_indices,unlabeled_indices = select_subset(labeled_indices=[],unlabeled_indices=list(range(datasize)),features=None,features_available=False,num_labels=n_labels,numclasses=numclasses,labels=labels,images=images,total_size=datasize,setting=args.setting,strat=strat,model=net)
            if args.setting=="BADGE" or args.setting=="US":
                X = list(np.array(train_dataset)[labeled_indices])
                Y = labels[labeled_indices]
                strat.update_data(labeled_dataset=list(np.array(labeled_dataset)[labeled_indices]),unlabeled_dataset=list(np.array(train_dataset)[unlabeled_indices]))
            #saving counter of selected subset
            counter_map["counter"] = counter(labeled_indices=labeled_indices,labels=labels)

            #save labeled and unlabeled indices
            np.save(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting), labeled_indices)
            np.save(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting), unlabeled_indices)
            np.save(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting+"_round1"), labeled_indices)
            np.save(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting+"_round1"), unlabeled_indices)

            #save images, labels files for SSL training
            l_images = images[labeled_indices]
            l_labels = labels[labeled_indices]
            u_images = images[unlabeled_indices]
            u_labels = np.zeros((len(unlabeled_indices),))-1 #dummy labels
            l_train_set = {"images":l_images, "labels":l_labels}
            u_train_set = {"images":u_images, "labels":u_labels}
            np.save(os.path.join(_EXP_DATA_DIR, "l_train_"+args.setting), l_train_set)
            np.save(os.path.join(_EXP_DATA_DIR, "u_train_"+args.setting), u_train_set)
            labeled_data = [[l_train_set['images'][i].astype(np.float32),l_train_set['labels'][i]] for i in range(len(l_train_set['images']))]
            labeled_x = torch.Tensor([np.array(l_train_set['images'][i].astype(np.float32)) for i in range(len(l_train_set['images']))])
            labeled_y = torch.Tensor([np.array([l_train_set['labels'][i]]).astype(np.int64) for i in range(len(l_train_set['images']))]).to(torch.long)
            labeled_dataset = MyDataset(data=labeled_x,targets=labeled_y,transform=transform_labeled)
            labeled_dataloader = DataLoader(labeled_dataset,sampler=train_sampler(labeled_dataset),batch_size=args.batch_size,num_workers=args.num_workers,drop_last=True)
            unlabeled_dataloader = [[u_train_set['images'][i].astype(np.float32)] for i in range(len(u_train_set['images']))]
            unlabeled_x = torch.Tensor([np.array(u_train_set['images'][i].astype(np.float32)) for i in range(len(u_train_set['images']))])
            unlabeled_y = torch.Tensor([np.array([u_train_set['labels'][i]]) for i in range(len(u_train_set['images']))]).to(torch.long)
            unlabeled_dataset = MyDataset(data=unlabeled_x,targets=unlabeled_y,transform=TransformFixMatch(mean=cifar10_mean,std=cifar10_std),labeled=False)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,sampler=train_sampler(unlabeled_dataset),batch_size=args.batch_size*args.mu,num_workers=args.num_workers,drop_last=True)

            print("------------------------First round SSL training---------------------------------")
            #training model
            args.alg = "supervised"
            args.total_steps = args.supervised_steps
            args.eval_step = args.supervised_evalstep
            args.epochs = math.ceil(args.total_steps / args.eval_step)
            test_accuracies = train(args,labeled_trainloader=labeled_dataloader,unlabeled_trainloader=unlabeled_dataloader,test_loader=test_loader,optimizer=optimizer,model=net,ema_model=ema_model,scheduler=scheduler)
            test_accs.append(test_accuracies)
            #subprocess.run(["python3","train.py","--dataset",args.dataset,"--alg","supervised","--output",exp_dir,"--validation",str(args.validation),"--root",args.root,"--nlabels",str(n_labels),"--gpudevice",args.gpudevice,"--setting",args.setting,"--run",str(args.run),"--round",str(cur_round),"--iteration",str(args.iterations), "--input_channels", str(args.input_channels)])
            cur_round+=1
            print("------------------------First round SSL training Done---------------------------------")
        elif cur_round == 2 and setting != "random":
            #Select B/numrounds number of points from SMI functions from all classes

            #Deleting previous model
            # prev_model_file = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-2)*(args.num_labeled)//(args.rounds))+ "_" + "supervised" + "_" + str(args.alg) + "_"
            # prev_model_file += "run"+str(args.run)
            # prev_model_file += "_round"+str(cur_round-2)
            # prev_model_file = os.path.join(exp_dir,prev_model_file+"_latest_model.pth")
            # if os.path.exists(prev_model_file):
            #     os.remove(prev_model_file)

            #Model retrieving
            for param in net.parameters():
                param.requires_grad = True
            # alg_cfg = config[args.alg]
            # optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])
            # mfile = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-1)*(args.num_labeled)//(args.rounds))+ "_" + "random" + "_" + "supervised" + "_"
            # mfile += "run"+str(args.run)
            # mfile += "_round"+str(cur_round-1)
            # dummy_exp_name = ""
            # dummy_exp_name += str(args.dataset) + "_"
            # dummy_exp_name += "labeled" + str(n_labels) + "_" + "random" + "_"
            # dummy_exp_name += str(args.alg) + "_"
            # dummy_exp_name += "run"+str(args.run)
            # dummy_exp_dir = os.path.join(args.output, dummy_exp_name)
            # if not os.path.exists(dummy_exp_dir):
            #     os.makedirs(dummy_exp_dir)
            # modelfile = os.path.join(dummy_exp_dir,mfile+"_latest_model.pth")
            # checkpoint = torch.load(modelfile)
            # model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            #Feature extraction
            train_dataset = [train_set['images'][i].astype(np.float32) for i in range(len(train_set['images']))]
            s = Strategy(labeled_dataset=train_dataset, unlabeled_dataset=np.array([]), net=net,nclasses=numclasses, args={'batch_size':50, 'device':device})
            #features = s.get_feature_embedding(dataset=train_dataset, unlabeled=True, layer_name="unit4")
            features = s.get_grad_embedding(dataset=train_dataset,predict_labels=True,grad_embedding_type='bias_linear')
            features = features.cpu().numpy()
            print("Shape of features: ",features.shape)
            
            #Subset selection using features obtained above
            #Loading unlabeled, labeled indices
            DUMMY_EXP_DATA_DIR = os.path.join(_DATA_DIR, args.dataset, str(args.num_labeled), "random")
            labeled_indices = list(np.load(os.path.join(DUMMY_EXP_DATA_DIR, "labeled_"+"random_round1"+".npy")))
            unlabeled_indices = list(np.load(os.path.join(DUMMY_EXP_DATA_DIR, "unlabeled_"+"random_round1"+".npy")))

            if args.setting=="BADGE" or args.setting=="US":
                #Update the model with the loaded unlabeled and labeled indices
                strat.update_data(labeled_dataset=list(np.array(labeled_dataset)[labeled_indices]),unlabeled_dataset=list(np.array(train_dataset)[unlabeled_indices]))

            #selecting new points
            print("------------------------Subset selection in process in "+ str(cur_round)+"th round---------------------------------")
            labeled_indices,unlabeled_indices = select_subset(labeled_indices=labeled_indices,unlabeled_indices=unlabeled_indices,features=features,features_available=True,num_labels=n_labels,numclasses=numclasses,labels=labels,images=images,total_size=datasize,setting=args.setting,strat=strat,model=net)
            if args.setting=="BADGE" or args.setting=="US":
                X = list(np.array(train_dataset)[labeled_indices])
                Y = labels[labeled_indices]
                strat.update_data(labeled_dataset=list(np.array(labeled_dataset)[labeled_indices]),unlabeled_dataset=list(np.array(train_dataset)[unlabeled_indices]))
            #saving counter of selected subset
            counter_map["counter"] = counter(labeled_indices=labeled_indices,labels=labels)

            #save labeled and unlabeled indices
            np.save(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting), labeled_indices)
            np.save(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting), unlabeled_indices)

            #save images, labels files for SSL training
            l_images = images[labeled_indices]
            l_labels = labels[labeled_indices]
            u_images = images[unlabeled_indices]
            u_labels = np.zeros((len(unlabeled_indices),))-1 #dummy labels
            l_train_set = {"images":l_images, "labels":l_labels}
            u_train_set = {"images":u_images, "labels":u_labels}
            np.save(os.path.join(_EXP_DATA_DIR, "l_train_"+args.setting), l_train_set)
            np.save(os.path.join(_EXP_DATA_DIR, "u_train_"+args.setting), u_train_set)
            labeled_data = [[l_train_set['images'][i].astype(np.float32),l_train_set['labels'][i]] for i in range(len(l_train_set['images']))]
            labeled_x = torch.Tensor([np.array(l_train_set['images'][i].astype(np.float32)) for i in range(len(l_train_set['images']))])
            labeled_y = torch.Tensor([np.array([l_train_set['labels'][i]]) for i in range(len(l_train_set['images']))]).to(torch.long)
            labeled_dataset = MyDataset(data=labeled_x,targets=labeled_y,transform=transform_labeled)
            labeled_dataloader = DataLoader(labeled_dataset,sampler=train_sampler(labeled_dataset),batch_size=args.batch_size,num_workers=args.num_workers,drop_last=True)
            unlabeled_dataloader = [[u_train_set['images'][i].astype(np.float32)] for i in range(len(u_train_set['images']))]
            unlabeled_x = torch.Tensor([np.array(u_train_set['images'][i].astype(np.float32)) for i in range(len(u_train_set['images']))])
            unlabeled_y = torch.Tensor([np.array([u_train_set['labels'][i]]) for i in range(len(u_train_set['images']))]).to(torch.long)
            unlabeled_dataset = MyDataset(data=unlabeled_x,targets=unlabeled_y,transform=TransformFixMatch(mean=cifar10_mean,std=cifar10_std),labeled=False)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,sampler=train_sampler(unlabeled_dataset),batch_size=args.batch_size*args.mu,num_workers=args.num_workers,drop_last=True)

            #training model
            print("------------------------" + str(cur_round)+ "th round SSL training---------------------------------")
            args.alg = "supervised"
            args.total_steps = args.supervised_steps
            args.eval_step = args.supervised_evalstep
            args.epochs = math.ceil(args.total_steps / args.eval_step)
            test_accuracies = train(args,labeled_trainloader=labeled_dataloader,unlabeled_trainloader=unlabeled_dataloader,test_loader=test_loader,optimizer=optimizer,model=net,ema_model=ema_model,scheduler=scheduler)
            test_accs.append(test_accuracies)
            cur_round += 1
        else:
            #Select B/numrounds number of points from SMI functions from all classes

            #Deleting previous model
            # prev_model_file = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-2)*(args.num_labeled)//(args.rounds))+ "_" + str(args.setting) + "_" + "supervised" + "_"
            # prev_model_file += "run"+str(args.run)
            # prev_model_file += "_round"+str(cur_round-2)
            # prev_model_file = os.path.join(exp_dir,prev_model_file+"_latest_model.pth")
            # if os.path.exists(prev_model_file) and cur_round!=3:
            #     os.remove(prev_model_file)

            # #Model retrieving
            # model = wrn.WRN(2, dataset_cfg["num_classes"], args.input_channels, transform_fn).to(device)
            for param in net.parameters():
                param.requires_grad = True
            # alg_cfg = config[args.alg]
            # optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])
            # mfile = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-1)*(args.num_labeled)//(args.rounds))+ "_" + str(args.setting) + "_" + "supervised" + "_"
            # mfile += "run"+str(args.run)
            # mfile += "_round"+str(cur_round-1)
            # modelfile = os.path.join(exp_dir,mfile+"_latest_model.pth")
            # checkpoint = torch.load(modelfile)
            # model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            #Feature extraction
            train_dataset = [train_set['images'][i].astype(np.float32) for i in range(len(train_set['images']))]
            s = Strategy(labeled_dataset=train_dataset, unlabeled_dataset=np.array([]), net=net,nclasses=numclasses, args={'batch_size':50, 'device':device})
            #features = s.get_feature_embedding(dataset=train_dataset, unlabeled=True, layer_name="unit4")
            features = s.get_grad_embedding(dataset=train_dataset,predict_labels=True,grad_embedding_type='bias_linear')
            features = features.cpu().numpy()
            print("Shape of features: ",features.shape)
            
            #Subset selection using features obtained above
            #Loading unlabeled, labeled indices
            labeled_indices = list(np.load(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting+".npy")))
            unlabeled_indices = list(np.load(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting+".npy")))

            #selecting new points
            print("------------------------Subset selection in process in "+ str(cur_round)+"th round---------------------------------")
            labeled_indices,unlabeled_indices = select_subset(labeled_indices=labeled_indices,unlabeled_indices=unlabeled_indices,features=features,features_available=True,num_labels=n_labels,numclasses=numclasses,labels=labels,images=images,total_size=datasize,setting=args.setting,strat=strat,model=net)
            if args.setting=="BADGE" or args.setting=="US":
                X = list(np.array(train_dataset)[labeled_indices])
                Y = labels[labeled_indices]
                strat.update_data(labeled_dataset=list(np.array(labeled_dataset)[labeled_indices]),unlabeled_dataset=list(np.array(train_dataset)[unlabeled_indices]))
            #saving counter of selected subset
            counter_map["counter"] = counter(labeled_indices=labeled_indices,labels=labels)

            #save labeled and unlabeled indices
            np.save(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting), labeled_indices)
            np.save(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting), unlabeled_indices)

            #save images, labels files for SSL training
            l_images = images[labeled_indices]
            l_labels = labels[labeled_indices]
            u_images = images[unlabeled_indices]
            u_labels = np.zeros((len(unlabeled_indices),))-1 #dummy labels
            l_train_set = {"images":l_images, "labels":l_labels}
            u_train_set = {"images":u_images, "labels":u_labels}
            np.save(os.path.join(_EXP_DATA_DIR, "l_train_"+args.setting), l_train_set)
            np.save(os.path.join(_EXP_DATA_DIR, "u_train_"+args.setting), u_train_set)
            labeled_data = [[l_train_set['images'][i].astype(np.float32),l_train_set['labels'][i]] for i in range(len(l_train_set['images']))]
            labeled_x = torch.Tensor([np.array(l_train_set['images'][i].astype(np.float32)) for i in range(len(l_train_set['images']))])
            labeled_y = torch.Tensor([np.array([l_train_set['labels'][i]]) for i in range(len(l_train_set['images']))]).to(torch.long)
            labeled_dataset = MyDataset(data=labeled_x,targets=labeled_y,transform=transform_labeled)
            labeled_dataloader = DataLoader(labeled_dataset,sampler=train_sampler(labeled_dataset),batch_size=args.batch_size,num_workers=args.num_workers,drop_last=True)
            unlabeled_dataloader = [[u_train_set['images'][i].astype(np.float32)] for i in range(len(u_train_set['images']))]
            unlabeled_x = torch.Tensor([np.array(u_train_set['images'][i].astype(np.float32)) for i in range(len(u_train_set['images']))])
            unlabeled_y = torch.Tensor([np.array([u_train_set['labels'][i]]) for i in range(len(u_train_set['images']))]).to(torch.long)
            unlabeled_dataset = MyDataset(data=unlabeled_x,targets=unlabeled_y,transform=TransformFixMatch(mean=cifar10_mean,std=cifar10_std),labeled=False)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,sampler=train_sampler(unlabeled_dataset),batch_size=args.batch_size*args.mu,num_workers=args.num_workers,drop_last=True)

            #training model
            print("------------------------" + str(cur_round)+ "th round SSL training---------------------------------")
            args.alg = 'supervised'
            args.total_steps = args.supervised_steps
            args.eval_step = args.supervised_evalstep
            args.epochs = math.ceil(args.total_steps / args.eval_step)
            test_accuracies = train(args,labeled_trainloader=labeled_dataloader,unlabeled_trainloader=unlabeled_dataloader,test_loader=test_loader,optimizer=optimizer,model=net,ema_model=ema_model,scheduler=scheduler)
            test_accs.append(test_accuracies)
            cur_round += 1
        temp_exp_dir = os.path.join(exp_dir,temp_exp_name)
        tempfile = os.path.join(temp_exp_dir,s_temp_exp_name+".json")
        curcounter_map = counter(labeled_indices,labels)
        curcounter = []
        for i in range(numclasses):
            if i in curcounter_map:
                curcounter.append(curcounter_map[i])
            else:
                curcounter.append(0)
        if cur_round==2:
            dummy_temp_exp_name = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-1)*(args.num_labeled)//(args.rounds))+ "_" + "random" + "_" + "supervised" + "_"
            dummy_temp_exp_name += "run"+str(args.run)
            dummy_temp_exp_name += "_round"+str(cur_round-1)
            s_dummy_temp_exp_name = ""+str(args.dataset)+"_"+"labeled" + str((cur_round-1)*(args.num_labeled)//(args.rounds))+ "_" + "random" + "_" + "supervised" + "_"
            s_dummy_temp_exp_name += "run"+str(args.run)
            s_dummy_temp_exp_name += "_round"+str(cur_round-1)
            sel_per_cls.append([int(i) for i in list(np.array(list(curcounter)))])
            dummy_exp_name = ""
            dummy_exp_name += str(args.dataset) + "_"
            dummy_exp_name += "labeled" + str(n_labels) + "_" + "random" + "_"
            dummy_exp_name += str(args.alg) + "_"
            dummy_exp_name += "run"+str(args.run)
            dummy_exp_dir = os.path.join(args.out, dummy_exp_name)
            if not os.path.exists(dummy_exp_dir):
                os.makedirs(dummy_exp_dir)
            dummy_temp_exp_dir = os.path.join(dummy_exp_dir,dummy_temp_exp_name)
            tempfile = os.path.join(dummy_temp_exp_dir,s_dummy_temp_exp_name+".json")
        else: 
            #counter,prev_counter)
            sel_per_cls.append([int(i) for i in list(np.array(list(curcounter))-np.array(list(prev_counter)))])
        logs["sel_per_cls"] = sel_per_cls
        logs["test_acc"] = test_accs
        jsonfile = exp_dir+"/"+exp_name+".json"
        with open(jsonfile,'w') as f:
            json.dump(logs,f)
        
    #Run model for 1 last time with whole bugdet labeled set
    labeled_indices = list(np.load(os.path.join(_EXP_DATA_DIR, "labeled_"+args.setting+".npy")))
    unlabeled_indices = list(np.load(os.path.join(_EXP_DATA_DIR, "unlabeled_"+args.setting+".npy")))

    l_images = images[labeled_indices]
    l_labels = labels[labeled_indices]
    u_images = images[unlabeled_indices]
    u_labels = np.zeros((len(unlabeled_indices),))-1 #dummy labels
    l_train_set = {"images":l_images, "labels":l_labels}
    u_train_set = {"images":u_images, "labels":u_labels}
    np.save(os.path.join(_EXP_DATA_DIR, "l_train_"+args.setting), l_train_set)
    np.save(os.path.join(_EXP_DATA_DIR, "u_train_"+args.setting), u_train_set)
    
    labeled_data = [[l_train_set['images'][i].astype(np.float32),l_train_set['labels'][i]] for i in range(len(l_train_set['images']))]
    labeled_x = torch.Tensor([np.array(l_train_set['images'][i].astype(np.float32)) for i in range(len(l_train_set['images']))])
    labeled_y = torch.Tensor([np.array([l_train_set['labels'][i]]) for i in range(len(l_train_set['images']))]).to(torch.long)
    labeled_dataset = MyDataset(data=labeled_x,targets=labeled_y,transform=transform_labeled)
    labeled_dataloader = DataLoader(labeled_dataset,sampler=train_sampler(labeled_dataset),batch_size=args.batch_size,num_workers=args.num_workers,drop_last=True)
    unlabeled_dataloader = [[u_train_set['images'][i].astype(np.float32)] for i in range(len(u_train_set['images']))]
    unlabeled_x = torch.Tensor([np.array(u_train_set['images'][i].astype(np.float32)) for i in range(len(u_train_set['images']))])
    unlabeled_y = torch.Tensor([np.array([u_train_set['labels'][i]]) for i in range(len(u_train_set['images']))]).to(torch.long)
    unlabeled_dataset = MyDataset(data=unlabeled_x,targets=unlabeled_y,transform=TransformFixMatch(mean=cifar10_mean,std=cifar10_std),labeled=False)
    unlabeled_dataloader = DataLoader(unlabeled_dataset,sampler=train_sampler(unlabeled_dataset),batch_size=args.batch_size*args.mu,num_workers=args.num_workers,drop_last=True)

    print("------------------------last round SSL training---------------------------------")
    args.total_steps = tot_steps
    args.eval_step = eval_step
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    test_accuracies = train(args,labeled_trainloader=labeled_dataloader,unlabeled_trainloader=unlabeled_dataloader,test_loader=test_loader,optimizer=optimizer,model=net,ema_model=ema_model,scheduler=scheduler)
    test_accs.append(test_accuracies)
    logs["test_acc"] = test_accs
    jsonfile = exp_dir+"/"+exp_name+".json"
    with open(jsonfile,'w') as f:
        json.dump(logs,f)
    #subprocess.run(["python3","train.py","--dataset",args.dataset,"--alg",args.alg,"--output",exp_dir,"--validation",str(args.validation),"--root",args.root,"--nlabels",str(n_labels),"--gpudevice",args.gpudevice,"--setting",args.setting,"--run",str(args.run),"--round",str(cur_round),"--iteration 500000", "--input_channels", str(args.input_channels)])


