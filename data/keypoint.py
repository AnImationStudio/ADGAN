import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform1, get_transformBP
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase) #person images
        self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K') #keypoints
        self.dir_DP = os.path.join(opt.dataroot, opt.phase + 'DPI') #DensePoseData
        self.dir_TEX = os.path.join(opt.dataroot, opt.phase + 'Tex') #DensePoseTexture
        self.dir_SP = opt.dirSem #semantic
        self.SP_input_nc = opt.SP_input_nc

        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)
        # self.transformBP = get_transformBP(opt) 
        self.transformBP = get_transform1(opt, 2)
        self.transformSP1 = get_transform1(opt, 0)

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.pairs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.pairs.append(pair)

        print('Loading data pairs finished ...')

    def __getitem__(self, index):


        while(True):
            if self.opt.phase == 'train':
                index = random.randint(0, self.size-1)

            P1_name, P2_name = self.pairs[index]
            P1_path = os.path.join(self.dir_P, P1_name) # person 1
            BP1_path = os.path.join(self.dir_DP, P1_name[:-4] + '.npz') # bone of person 1
            # BP1_path = os.path.join(self.dir_K, P1_name + '.npy') # bone of person 1

            # person 2 and its bone
            P2_path = os.path.join(self.dir_P, P2_name) # person 2
            BP2_path = os.path.join(self.dir_DP, P2_name[:-4] + '.npz') # bone of person 2
            # BP2_path = os.path.join(self.dir_K, P2_name + '.npy') # bone of person 2

            SP1_path = os.path.join(self.dir_TEX, P2_name[:-4] + '.npz')

            if(os.path.exists(BP1_path) and os.path.exists(BP2_path) and os.path.exists(SP1_path)):
                break


        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')

        BP1_img = np.load(BP1_path)['arr_0'] # h, w, c
        BP2_img = np.load(BP2_path)['arr_0'] 
        # BP1_img = np.load(BP1_path) # h, w, c
        # BP2_img = np.load(BP2_path)

        # use flip
        if self.opt.phase == 'train' and self.opt.use_flip:
            # print ('use_flip ...')
            flip_random = random.uniform(0,1)
            
            if flip_random > 0.5:
                # print('fliped ...')
                P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
                P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)

                BP1_img = np.array(BP1_img[:, ::-1, :]) # flip
                BP2_img = np.array(BP2_img[:, ::-1, :]) # flip

            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

        else:
            BP1 = torch.from_numpy(BP1_img).float() #h, w, c
            BP1[:,:,0] = BP1[:,:,0]/24.0
            # print("BP Min ", torch.amin(BP1, dim=1), torch.amax(BP1, dim=1))
            BP1 = BP1.transpose(2, 0) #c,w,h
            BP1 = BP1.transpose(2, 1) #c,h,w 

            BP2 = torch.from_numpy(BP2_img).float()
            BP2[:,:,0] = BP2[:,:,0]/24.0
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 

            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)
        BP1 = self.transformBP(BP1)
        BP2 = self.transformBP(BP2)

        # print(BP1.shape)
        # print("BP Min ", torch.amin(BP1, dim=(1,2)), torch.amax(BP1, dim=(1,2)))

        # segmentation
        # SP1_name = self.split_name(P1_name, 'semantic_merge3')
        # SP1_path = os.path.join(self.dir_SP, SP1_name)
        # SP1_path = SP1_path[:-4] + '.npy'
        
        # SP1_data = np.load(SP1_path)
        # SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
        # for id in range(self.SP_input_nc):
        #     SP1[id] = (SP1_data == id).astype('float32')

        SP1 = np.load(SP1_path)['arr_0']
        SP1 = torch.from_numpy(SP1).float()        
        SP1 = SP1.transpose(3, 1) #c,w,h
        SP1 = SP1.transpose(3, 2) #c,h,w 
        # SP1 = self.transformSP1(SP1)

        # print("Input dimensions ", P1.shape, P2.shape, BP1.shape, BP2.shape,
        #     SP1.shape)
        # print("Data Min Max ", torch.amin(P1, dim=(1,2)), torch.amax(P1, dim=(1,2)),
        #     torch.amin(BP1, dim=(1,2)), torch.amax(BP1, dim=(1,2)),
        #     torch.amin(SP1, dim=(1,2)), torch.amax(SP1, dim=(1,2)))

        return {'P1': P1, 'BP1': BP1, 'SP1': SP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': P2_name}

                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self,str,type):
        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2]+'_'+id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4]+'_'+pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

