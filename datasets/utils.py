import torch
import os
import json
import numpy as np
from scipy import io
from PIL import Image
from torch.utils.data import Dataset
import cv2
import time
import tqdm
import random


def normalize(x):
    '''
    scale the vector length to 1.
    params:
        x: torch tensor, shape "[...,vector_dim]"
    '''
    norm = torch.sqrt(torch.sum(torch.pow(x,2),-1,keepdims=True))
    return x / (norm+1e-6)

def find_all_files(path,filter='sar.jpg'):
    total = []
    if not os.path.isdir(path):
        if path.endswith(filter):
            total.append(path)
    else:
        subs = os.listdir(path)
        for sub in subs:
            total.extend(find_all_files(os.path.join(path,sub),filter))
    return total

# def compute_confidence_matrix(lf0,lf1):
#     d = lf0.shape[2]
#     lf0 = lf0 / d
#     lf1 = lf1 / d
#     similarity_matrix = torch.matmul(lf0,lf1.transpose(1,2))
#     confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2)
#     return confidence_matrix
def random_rotation2(img,src, qc, max_degree=4,borderValue=None):
    src = src[qc[1]-70: qc[1]-70+460, qc[0]-70:qc[0]-70+460]
    s_h, s_w = src.shape[:2]
    i_h, i_w = img.shape[:2]
    img=cv2.copyMakeBorder(img, 0, 320-i_h ,0, 320-i_w,
                                 cv2.BORDER_CONSTANT,value = (0,0,0))
    src=cv2.copyMakeBorder(src,0, 460-s_h ,0, 460-s_w,
                                 cv2.BORDER_CONSTANT,value = (0,0,0))

    i_h,i_w = img.shape[:2]
    s_h,s_w = src.shape[:2]
    degree = np.random.rand(1) * max_degree * 2 - max_degree
    matRotate = cv2.getRotationMatrix2D((s_h*0.5, s_w*0.5),float(degree),1)
    if borderValue is None:
        borderValue = (127.5,127.5,127.5)
    src0 = cv2.warpAffine(src, matRotate, (s_w,s_h), borderValue=borderValue)
    src = src0[(s_h-i_h)//2:(s_h-i_h)//2+320, (s_w-i_w)//2:(s_w-i_w)//2+320]

    #degree = np.random.rand(1) * max_degree * 2 - max_degree
    #print(matRotate, "===>", matRotate1, '\n')
    if borderValue is None:
        borderValue = (127.5,127.5,127.5)
    matRotate1 = cv2.getRotationMatrix2D((i_h*0.5, i_w*0.5),float(degree),1)
    img = cv2.warpAffine(img, matRotate1, (i_w,i_h), borderValue=borderValue)
    #cv2.imshow("img1", img)
    #cv2.imshow("img2", src0)
    #cv2.imshow("img", src)
    #cv2.waitKey()
    return src, matRotate1

def random_rotation(img,max_degree=4,borderValue=None):
    h,w = img.shape[:2]
    degree = np.random.rand(1) * max_degree * 2 - max_degree
    matRotate = cv2.getRotationMatrix2D((h*0.5, w*0.5),float(degree),1)
    if borderValue is None:
        # borderValue = (np.random.randint(0,255),) * 3
        borderValue = (127.5,127.5,127.5)
    img = cv2.warpAffine(img, matRotate, (w,h),borderValue=borderValue)
    return img,matRotate



def compute_confidence_matrix(lf0,lf1,use_arc=False):
    if use_arc:
        lf0 = normalize(lf0)
        lf1 = normalize(lf1)
        similarity_matrix = torch.matmul(lf0,lf1.transpose(1,2)) * 64
        confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2)
    else:
        _d = lf0.shape[-1]
        lf0 = lf0 / _d
        lf1 = lf1 / _d
        similarity_matrix = torch.matmul(lf0,lf1.transpose(1,2)) * 10
        confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2) 
    return confidence_matrix

def load_weights(model, model_path=None):
    if not os.path.exists(model_path):
        print("%s not exist, start training from stratch..."%model_path)
        return -1
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("restored from %s"%model_path)

def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid                    

def random_gauss_noise(image,mean=0,var=0.1):
    image = np.array(image/255, dtype=float)
    var_scale = np.random.rand(1)
    noise = np.random.normal(mean, var*var_scale, (image.shape[0],image.shape[1],1))
    noise = np.tile(noise,(1,1,3))
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out*255)
    return out

def blur_image(image,kernel_size=7):
    img = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    return img



def draw_match(match_mask,query,refer,homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts[::16]:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    return out_img


def draw_match2(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        if np.abs(x1-512-x0-x) < 5 and np.abs(y1-y-y0) < 5:
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

