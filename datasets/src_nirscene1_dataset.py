import cv2
import os
from torch.utils.data import Dataset,DataLoader
import random
import torch
import numpy as np


class srcnirscene1Dataset(Dataset):
    def __init__(self,data_file,size=(320,320),stride=8):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = size
        self.stride = stride # for generating gt-mask needed to compute local-feature loss
        self.query_pts = self._make_query_pts()
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)

    def _read_file_paths(self,data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only"%data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        opt, sar  = self.train_data[index].strip('\n').split(' ')
        opt_img_path = os.path.join(os.path.dirname(self.data_file), '', opt)
        opt_img = cv2.imread(opt_img_path.replace('stage1_', ''))
        opt_img = cv2.cvtColor(opt_img,cv2.COLOR_BGR2RGB)
        h, w ,c  = opt_img.shape
        w_ratio = w / 512
        h = int((h / w_ratio) // 32) * 32
        opt_img = cv2.resize(opt_img, (512, h))
        #opt_img = opt_img[:h-64, :]
        #opt_img = cv2.copyMakeBorder(opt_img, 64, 0, 0,0, cv2.BORDER_CONSTANT, value=(0,0,0))

        sar_img_path = os.path.join(os.path.dirname(self.data_file), '', sar)
        sar_img = cv2.imread(sar_img_path.replace('stage1_', ''))
        sar_img = cv2.cvtColor(sar_img,cv2.COLOR_BGR2RGB)
        sar_img = cv2.resize(sar_img, (512, h))
        #sar_img = sar_img[:h-64, :]
        #sar_img = cv2.copyMakeBorder(sar_img, 64, 0, 0,0, cv2.BORDER_CONSTANT, value=(0,0,0))

        #op_h, op_w, c = sar_img.shape
        #print(op_h // h_ratio, h_ratio, int(x) // w_ratio, int(y) // h_ratio, 512 // w_ratio, 512 // h_ratio)
        #sar_img = cv2.resize(sar_img, (int(512 // h_ratio), int(512 // w_ratio)))
        #print(sar_img.shape)

        #query,refer,Mr,Mq, qc, rc = self._generate_ref(opt_img, sar_img)

        # dropout query
        #label_matrix = self._generate_label(Mr,Mq, qc, rc, (0,0)) #400x400

        #cv2.imshow("query:", query)
        #cv2.imshow("refer:", refer)
        #cv2.waitKey()
        query = sar_img.transpose(2,0,1)
        refer = opt_img.transpose(2,0,1)

        query = ((query / 255.0) - self.mean) / self.std
        refer = ((refer / 255.0) - self.mean) / self.std

        sample = {
            "refer":refer,
            "query":query,
            #"gt_matrix":label_matrix,
            # "M": M,
            # "Mr": Mr,
            # "Mq": Mq
        }
        return sample

    def _generate_ref(self,refer, query):
        """
        通过sar和optical找到相对应的映射关系矩阵
        """

        #refer,M = random_place(ref, query, x, y, w, h)
        # refer = ref_back.copy()

        #cv2.imshow("before:", query)


        crop_query, crop_M_query, qc = self._random_crop(query)

        query, Mq = self._aug_img(crop_query) # 320x320x3, 3x3

        #cv2.imshow("after:", query)
        #cv2.waitKey()

        # np_img = np.ones_like(query) * 128
        # scale = 0.5
        # np_img[:int(self.size[1]*scale),:int(self.size[0]*scale)] = cv2.resize(query,None,fx=scale,fy=scale)
        # query = np_img
        Mq = np.matmul(Mq, crop_M_query)
        crop_refer, crop_M_refer, rc = self._random_crop(refer)
        refer, Mr = self._aug_img(crop_refer)
        Mr = np.matmul(Mr, crop_M_query)


        return query,refer, Mr,Mq, qc, rc

    def _generate_label(self,Mr,Mq, qc, rc, coor, drop_mask=True):
        """
        M random_place
        Mr aug_refer
        Mq aug_query
        """
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        label = np.zeros((ncols*nrows,ncols*nrows))

        Mq_inv = np.linalg.inv(Mq)
        src_pts = np.matmul(Mq_inv, self.query_pts.T) #self.query_pts (3x400) , shape:20x20x3, 变换位置
        mask0 = (0<= src_pts[0,:]) & (src_pts[0, :] < 320) & (0 <= src_pts[1, :]) & (src_pts[1, :] < 320)

        #sar原图平移
        trans_M = np.array([
            [1, 0, coor[0]],
            [0, 1, coor[1]],
            [0, 0, 1]
            ])
        refer_pts = np.matmul(trans_M, src_pts)
        #平移得到sar和opt对其
        trans_M = np.array([
            [1, 0, qc[0]-rc[0]],
            [0, 1, qc[1]-rc[1]],
            [0, 0, 1]
            ])
        refer_pts = np.matmul(trans_M, refer_pts)
        #opt原图裁剪
        refer_pts = np.matmul(Mr, refer_pts)
        mask1 = (0<= refer_pts[0,:]) & (refer_pts[0, :] < 320) & (0 <= refer_pts[1, :]) & (refer_pts[1, :] < 320)

        mask =  mask1

        match_index = np.int32(refer_pts[0, :]//self.stride + (refer_pts[1, :]//self.stride)*ncols)
        indexes = np.arange(nrows*ncols)[mask]
        for index in indexes:
            label[index][match_index[index]] = 1
        return label

    def _make_query_pts(self):
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        half_stride = (self.stride-1) / 2
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        xs = np.tile(xs[np.newaxis,:],(nrows,1))
        ys = np.tile(ys[:,np.newaxis],(1,ncols))
        ones = np.ones((nrows,ncols,1))
        grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis],ones],axis=-1)
        grid[:,:,:2] = grid[:,:,:2] * self.stride + half_stride  #(0:20, 0:20, 1) , shape:20x20x3
        return grid.reshape(-1,3) # (nrows*ncols , 3)

    def _random_flag(self,thresh=-1):
        return np.random.rand(1) < thresh

    def _random_crop(self, img):
        h, w, c = img.shape

        #matrix = np.eye(3)
        x, y = random.randint(0, min(h-320, w-320)), random.randint(0, min(h-320, w-320))
        #x,y = 0, 3
        img = img[y:320+y, x:320+x]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
            ])
        #img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)


    def _aug_img(self,img):
        h,w = img.shape[:2]
        matrix = np.eye(3)
        if self._random_flag():
            img = img[:,::-1,...].copy() # horizontal flip
            fM = np.array([
                [-1,0,w-1],
                [0,1,0],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(fM,matrix)

        if self._random_flag():
            img = img[::-1,:,...].copy() # vertical flip
            vfM = np.array([
                [1,0,0],
                [0,-1,h-1],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(vfM,matrix)

        if self._random_flag():
            img = change_lightness_contrast(img) # change light

        if self._random_flag():
            h,s,v = np.random.rand(3)/2.5 - 0.2
            img = random_distort_hsv(img,h,s,v)

        if self._random_flag():
            img = random_gauss_noise(img)

        if self._random_flag():
            img = random_mask(img)

        if self._random_flag():
            img,sh,sw = random_jitter(img,max_jitter=0.3)
            jM = np.array([
                [1,0,sw],
                [0,1,sh],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(jM,matrix)

        if self._random_flag(0):
            img,rM = random_rotation(img,max_degree=45)
            rM = np.concatenate([rM,np.array([[0,0,1]],np.float32)])
            matrix = np.matmul(rM,matrix)

        if self._random_flag():
            kernel = random.choice([3,5,7])
            img = blur_image(img,kernel)
        return img,matrix

    def __len__(self):
        return len(self.train_data)

if __name__ == "__main__":
    from utils import _transform_inv,draw_match
    size = (320,320)
    dataloader = DataLoader(
        nirscene1Dataset("/home/ly/data/dataset/nirscene1/train.txt",size=size),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    print(len(dataloader))
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)
    check_index = 0
    while 1:
        for sample in dataloader:
            query,refer,label_matrix = sample["query"],sample["refer"],sample["gt_matrix"]
            query0 = query.detach().cpu().numpy()[check_index]
            refer0 = refer.detach().cpu().numpy()[check_index]
            label_matrix0 = label_matrix.detach().cpu().numpy()[check_index]
            query1 = query.detach().cpu().numpy()[check_index+1]
            refer1 = refer.detach().cpu().numpy()[check_index+1]
            label_matrix1 = label_matrix.detach().cpu().numpy()[check_index+1]

            sq0 = _transform_inv(query0,mean,std)
            sr0 = _transform_inv(refer0,mean,std)
            out0 = draw_match(label_matrix0>0,sq0,sr0).squeeze()
            sq1 = _transform_inv(query1,mean,std)
            sr1 = _transform_inv(refer1,mean,std)
            out1 = draw_match(label_matrix1>0,sq1,sr1).squeeze()
            cv2.imshow("match_img0",out0)
            cv2.imshow("match_img1",out1)
            cv2.waitKey()


def build_src_nir(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = srcnirscene1Dataset(
        train_data_file,
        size=(320, 320),
        stride=8)
    test_data = srcnirscene1Dataset(
        test_data_file,
        size=(320, 320),
        stride=8)

    return train_data, test_data


