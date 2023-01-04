import cv2
import os
from torch.utils.data import Dataset,DataLoader
import random
import torch
import numpy as np
try:
    from .utils import *
except:
    from utils import *
from skimage import io, color


class RotateNIRDataset(Dataset):
    def __init__(self,data_file,size=(320,320),stride=8, aug=True):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = size
        self.aug = aug
        self.stride = stride # for generating gt-mask needed to compute local-feature loss
        self.stride_16x=16
        self.query_pts = self._make_query_pts()
        self.query_pts_16x = self._make_query_pts_16x()
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)

    def _read_file_paths(self,data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only"%data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        opt, sar  = self.train_data[index].strip('\n').split(' ')
        opt_img_path = os.path.join(os.path.dirname(self.data_file), '', opt)
        opt_img = io.imread(opt_img_path.replace('stage1_', ''))
        #opt_img = color.rgba2rgb(opt_img)
        h, w ,c  = opt_img.shape

        sar_img_path = os.path.join(os.path.dirname(self.data_file), '', sar)
        sar_img = io.imread(sar_img_path.replace('stage1_', ''))
        sar_img = cv2.cvtColor(sar_img,cv2.COLOR_GRAY2RGB)

        #query,refer,Mr,Mq, qc, rc = self._generate_ref(opt_img, sar_img)

        # dropout query
        #label_matrix = self._generate_label(Mr,Mq, qc, rc, (int(0), int(0))) #400x400
        #label_matrix_16x = self._generate_label_16x(Mr,Mq, qc, rc, (int(0), int(0))) #400x400

        #cv2.imshow("query:", query)
        #cv2.imshow("refer:", refer)
        #cv2.waitKey()
        print(query.shape, refer.shape)
        query = query.transpose(2,0,1)
        refer = refer.transpose(2,0,1)

        query = ((query / 255.0) - self.mean) / self.std
        refer = ((refer / 255.0) - self.mean) / self.std

        sample = {
            "refer":refer,
            "query":query,
        }
        return sample

    def _generate_ref(self,refer, query):
        """
        通过sar和optical找到相对应的映射关系矩阵
        """


        if random.sample([0,1,2], 1)[0] != 0 and self.aug is True:
            crop_query, crop_M_query, qc = self._random_crop(query)
            query, Mq = self._aug_img(crop_query, query, qc) # 320x320x3, 3x3
            Mq = np.matmul(Mq, crop_M_query)

            crop_refer, crop_M_refer, rc = self._random_crop(refer)
            refer, Mr = self._aug_img(crop_refer, refer, rc)
            Mr = np.matmul(Mr, crop_M_refer)
        else:
            crop_query, crop_M_query, qc = self._random_crop2(query)
            query, Mq = self._aug_img(crop_query, query, qc, -1) # 320x320x3, 3x3
            Mq = np.matmul(Mq, crop_M_query)

            crop_refer, crop_M_refer, rc = self._random_crop2(refer)
            refer, Mr = self._aug_img(crop_refer, refer, rc, -1)
            Mr = np.matmul(Mr, crop_M_refer)


        return query,refer, Mr,Mq, qc, rc

    def _generate_label_16x(self,Mr,Mq, qc, rc, coor, drop_mask=True):
        """
        M random_place
        Mr aug_refer
        Mq aug_query
        """
        ncols, nrows = self.size[0] // self.stride_16x, self.size[1] // self.stride_16x
        label = np.zeros((ncols*nrows,ncols*nrows)) #(1600, 1600)

        Mq_inv = np.linalg.inv(Mq)
        src_pts = np.matmul(Mq_inv, self.query_pts_16x.T) #self.query_pts (3x1600) , shape:40x40x3, 变换位置
        mask0 = (0<= src_pts[0,:]) & (src_pts[0, :] < 320) & (0 <= src_pts[1, :]) & (src_pts[1, :] < 320)

        #sar原图平移
        trans_M = np.array([
            [1, 0, coor[0]],
            [0, 1, coor[1]],
            [0, 0, 1]
            ])
        refer_pts = np.matmul(trans_M, src_pts)
        #平移得到sar和opt对其
        trans_M1 = np.array([
            [1, 0, qc[0]],
            [0, 1, qc[1]],
            [0, 0, 1]
            ])
        trans_M2 = np.array([
            [1, 0, qc[0]-rc[0]],
            [0, 1, qc[1]-rc[1]],
            [0, 0, 1]
            ])
        trans_M = np.matmul(trans_M2, trans_M1)
        trans_M3 = np.array([
            [1, 0, -rc[0]],
            [0, 1, -rc[1]],
            [0, 0, 1]
            ])
        trans_M = np.matmul(trans_M3, trans_M)
        refer_pts = np.matmul(trans_M, refer_pts)
        #opt原图裁剪
        refer_pts = np.matmul(Mr, refer_pts)
        mask1 = (0<= refer_pts[0,:]) & (refer_pts[0, :] < 320) & (0 <= refer_pts[1, :]) & (refer_pts[1, :] < 320)

        mask =  mask1 #(1600,)

        match_index = np.int32(refer_pts[0, :]//self.stride_16x + (refer_pts[1, :]//self.stride_16x)*ncols) #(1600,)
        indexes = np.arange(nrows*ncols)[mask]
        for index in indexes:
            label[index][match_index[index]] = 1
        return label

    def _generate_label(self,Mr,Mq, qc, rc, coor, drop_mask=True):
        """
        M random_place
        Mr aug_refer
        Mq aug_query
        """
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        label = np.zeros((ncols*nrows,ncols*nrows)) #(1600, 1600)

        Mq_inv = np.linalg.inv(Mq)
        src_pts = np.matmul(Mq_inv, self.query_pts.T) #self.query_pts (3x1600) , shape:40x40x3, 变换位置
        mask0 = (0<= src_pts[0,:]) & (src_pts[0, :] < 320) & (0 <= src_pts[1, :]) & (src_pts[1, :] < 320)

        #sar原图平移
        trans_M = np.array([
            [1, 0, coor[0]],
            [0, 1, coor[1]],
            [0, 0, 1]
            ])
        refer_pts = np.matmul(trans_M, src_pts)
        #平移得到sar和opt对其
        trans_M1 = np.array([
            [1, 0, qc[0]],
            [0, 1, qc[1]],
            [0, 0, 1]
            ])
        trans_M2 = np.array([
            [1, 0, qc[0]-rc[0]],
            [0, 1, qc[1]-rc[1]],
            [0, 0, 1]
            ])
        trans_M = np.matmul(trans_M2, trans_M1)
        trans_M3 = np.array([
            [1, 0, -rc[0]],
            [0, 1, -rc[1]],
            [0, 0, 1]
            ])
        trans_M = np.matmul(trans_M3, trans_M)
        refer_pts = np.matmul(trans_M, refer_pts)
        #opt原图裁剪
        refer_pts = np.matmul(Mr, refer_pts)
        mask1 = (0<= refer_pts[0,:]) & (refer_pts[0, :] < 320) & (0 <= refer_pts[1, :]) & (refer_pts[1, :] < 320)

        mask =  mask1 #(1600,)

        match_index = np.int32(refer_pts[0, :]//self.stride + (refer_pts[1, :]//self.stride)*ncols) #(1600,)
        indexes = np.arange(nrows*ncols)[mask]
        for index in indexes:
            label[index][match_index[index]] = 1
        return label


    def _make_query_pts_16x(self):
        ncols, nrows = self.size[0] // self.stride_16x, self.size[1] // self.stride_16x
        half_stride = (self.stride_16x-1) / 2
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        xs = np.tile(xs[np.newaxis,:],(nrows,1))
        ys = np.tile(ys[:,np.newaxis],(1,ncols))
        ones = np.ones((nrows,ncols,1))
        grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis],ones],axis=-1)
        grid[:,:,:2] = grid[:,:,:2] * self.stride_16x + half_stride  #(0:20, 0:20, 1) , shape:20x20x3
        return grid.reshape(-1,3) # (nrows*ncols , 3)

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
        x, y = random.randint(70, max(w-460, 70)), random.randint(70, max(h-460, 70))
        #x,y = 70, 70
        #x,y = 0, 3
        img = img[y:320+y, x:320+x]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
            ])
        #img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)

    def _random_crop2(self, img):
        h, w, c = img.shape

        #matrix = np.eye(3)
        x, y = random.randint(0, w-320), random.randint(0, h-320)
        #x,y = 0, 3
        img = img[y:320+y, x:320+x]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
            ])
        #img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)

    def _aug_img(self,img, src, qc, aug=1):
        h,w = img.shape[:2]
        matrix = np.eye(3)
 
        if self._random_flag(aug):
            img,rM = random_rotation2(img,src, qc, max_degree=60)
            #img,rM = random_rotation(img, max_degree=45)
            rM = np.concatenate([rM,np.array([[0,0,1]],np.float32)])
            matrix = np.matmul(rM,matrix)

        if self._random_flag():
            kernel = random.choice([1, 3,5,7])
            img = blur_image(img,kernel)

        if self._random_flag(aug*0.2):
            img = img[:,::-1,...].copy() # horizontal flip
            fM = np.array([
                [-1,0,w-1],
                [0,1,0],
                [0,0,1]
            ],np.float32)
            matrix = np.matmul(fM,matrix)

        if self._random_flag(aug * 0.2):
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


        return img,matrix

            
    def __len__(self):
        return len(self.train_data)



def build_Rotate_NIR(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = RotateNIRDataset(
        train_data_file,
        size=(320, 320),
        stride=8,
        aug=True)
    test_data = RotateNIRDataset(
        test_data_file,
        size=(320, 320),
        stride=8,
        aug=False)

    return train_data, test_data



if __name__ == "__main__":
    from utils import _transform_inv,draw_match
    size = (320,320)
    dataloader = DataLoader(
        RotateNIRDataset("/first_disk/TopKWindows/Femit/Femit_dataset/nirscene1/train.txt",size=size, aug=True),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    print(len(dataloader))
    mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)
    check_index = 0
    num = 0
    while 1:
        for sample in dataloader:
            query,refer,label_matrix, label_matrix_16x = sample["query"],sample["refer"],sample["gt_matrix"], sample['gt_matrix_16x']
            query0 = query.detach().cpu().numpy()[check_index]
            refer0 = refer.detach().cpu().numpy()[check_index]
            label_matrix0 = label_matrix.detach().cpu().numpy()[check_index]
            query1 = query.detach().cpu().numpy()[check_index+1]
            refer1 = refer.detach().cpu().numpy()[check_index+1]
            label_matrix1 = label_matrix.detach().cpu().numpy()[check_index+1]

            label_matrix0_16x = label_matrix_16x.detach().cpu().numpy()[check_index]
            label_matrix1_16x = label_matrix_16x.detach().cpu().numpy()[check_index+1]
            sq0 = _transform_inv(query0,mean,std)
            sr0 = _transform_inv(refer0,mean,std)
            out0 = draw_match(label_matrix0>0,sq0,sr0).squeeze()
            out2 = draw_match_16x(label_matrix0_16x>0,sq0,sr0).squeeze()
            sq1 = _transform_inv(query1,mean,std)
            sr1 = _transform_inv(refer1,mean,std)
            out1 = draw_match(label_matrix1>0,sq1,sr1).squeeze()
            out3 = draw_match_16x(label_matrix1_16x>0,sq1,sr1).squeeze()
            cv2.imwrite(f"images/match_img0_{num}.jpg",out0)
            cv2.imwrite(f"images/match_img1_{num}.jpg",out1)
            cv2.imwrite(f"images/match_img2_{num}.jpg",out2)
            cv2.imwrite(f"images/match_img3_{num}.jpg",out3)
            num = num+ 1

