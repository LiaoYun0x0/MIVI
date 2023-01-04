import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.einops import rearrange, repeat
import cv2
import pydegensac
import random

def mask_border(m, b, v):
    m[:, :b]=v
    m[:, :, :b]=v
    m[:, :, :, :b]=v
    m[:, :, :, :, :b]=v
    m[:, -b:0]=v
    m[:, :, -b:0]=v
    m[:, :, :, -b:0]=v
    m[:, :, :, :, -b:0]=v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd]= v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, h1-bd:] = v

def cal_reproj_dists(p1s, p2s, homography):
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist

def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index-1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs)

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid 

def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

def get_mkpts(matches, query, refer, patch_size=8):
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([x0, y0]))
        refer_pts.append(torch.Tensor([x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts

def batch_get_mkpts(matches, query, refer, patch_size=8):
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([pt[0], x0, y0]))
        refer_pts.append(torch.Tensor([pt[0], x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 3).cuda()
        refer_pts = torch.Tensor(0, 3).cuda()

    return query_pts, refer_pts

def batch_get_mkpts_16x(matches, query, refer, patch_size=16):
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    x0 = patch_size/2 + (matches[:, 1] % qcols) * patch_size
    y0 = patch_size/2 + torch.div(matches[:, 1], qcols) * patch_size
    x1 = patch_size/2 + (matches[:, 2] % qcols) * patch_size
    y1 = patch_size/2 + torch.div(matches[:, 2], qcols) * patch_size
    query_pts = torch.cat((matches[:, 0].unsqueeze(1), x0.unsqueeze(1), y0.unsqueeze(1)), 1)
    refer_pts = torch.cat((matches[:, 0].unsqueeze(1), x1.unsqueeze(1), y1.unsqueeze(1)), 1)
    if len(query_pts) < 0:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts

def draw_match_kpts(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=1e3):
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 640
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-640-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        pts = random.sample([(q_pts[i], r_pts[i]) for i in range(query_pts.shape[0]) if mask[i]], 5)
        for i in range(len(pts)):
            cv2.rectangle(out_img, (int(pts[i][0][0]-8),int(pts[i][0][1]-8)), (int(pts[i][0][0]+8),int(pts[i][0][1]+8)), (0, 0, 255), 1)
            cv2.circle(out_img, (int(pts[i][0][0]),int(pts[i][0][1])),2, (0, 255, 255), -1)
            cv2.rectangle(out_img, (int(pts[i][1][0]-8),int(pts[i][1][1]-8)), (int(pts[i][1][0]+8),int(pts[i][1][1]+8)), (0, 0, 255), 1)
            cv2.circle(out_img, (int(pts[i][1][0]),int(pts[i][1][1])), 2,(0, 255, 255), -1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    return out_img

def draw_match_se(mkpts0, mkpts1, query,refer, x, y, homo_filter=True,patch_size=8, th=10):
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1-wq-x0-x) < th and np.abs(y1-y-y0) < th:
            q_pts.append([x0,y0])
            r_pts.append([x1,y1])
            x_mean.append(x1-wq-x0-x)
            y_mean.append(y1-y-y0)

    if len(x_mean) != 0:
        x_means = int(np.mean(x_mean))
        y_means = int(np.mean(y_mean))
    else:
        x_means,y_means = 0,0

    query_pts, refer_pts = q_pts, r_pts
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

def eval_mma(match_mask, mkpts0, mkpts1, query, refer, i_err, thres=[1,2,3,4,5,6,7,8,9,10],patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()

    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts

    q_pts, r_pts = get_gt_match(match_mask)
    if q_pts.shape[0]== 0:
        return i_err, 0
    H_gt, inliers = cv2.findHomography(q_pts, r_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1

def eval_homography(match_mask, mkpts0, mkpts1, query, refer,patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h,w = query.shape[:2]

    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1],match_mask.shape[0])
        _pts = grid[match_mask]
        #out_img = np.concatenate([query,refer],axis=1).copy()
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size/2 + (pt[1] % qcols) * patch_size
            y0 = patch_size/2 + (pt[1] // qcols) * patch_size
            x1 = patch_size/2 + (pt[0] % rcols) * patch_size
            y1 = patch_size/2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0,y0])
            refer_pts.append([x1,y1])
            # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
        query_pts = np.asarray(query_pts,np.float32)
        refer_pts = np.asarray(refer_pts,np.float32)

        return query_pts, refer_pts
    q_pts, r_pts = get_gt_match(match_mask)
    if q_pts.shape[0]== 0:
        return np.nan
    if mkpts0.shape[0] == 0:
        return np.nan
    H_gt, inliers = cv2.findHomography(q_pts, r_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thres)
    if H_gt is None:
        return np.nan
    corners = np.array([[0,0,1],[0,w-1,1],[h-1,0,1],[h-1,w-1,1]])
    H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thres)

    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    if H_pred is None:
        return np.nan
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    return corner_dist

def eval_src_mma(mkpts0, mkpts1, query, refer, i_err, thres=[1,2,3,4,5,6,7,8,9,10],patch_size=8, ransac_thres=3):
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    if len(mkpts1) >3:
        H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)
        if len([mkpts0[x, :] for x in np.where(inliers>0)[0]]) > 0:
            mkpts0 = np.stack([mkpts0[x, :] for x in np.where(inliers>0)[0]])
            mkpts1= np.stack([mkpts1[x, :] for x in np.where(inliers>0)[0]])
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])

    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1

def eval_src_homography(mkpts0, mkpts1, query, refer,patch_size=8, ransac_thres=3):
    np.set_printoptions(suppress=True)
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h,w = query.shape[:2]
    H_gt = np.array([[1,0,0],[0,1,0],[0,0,1]])
    if mkpts0.shape[0] < 4:
        return np.nan

    corners = np.array([[0,0,1],[0,w-1,1],[h-1,0,1],[h-1,w-1,1]])
    H_pred, inliers = pydegensac.findHomography(mkpts0, mkpts1, ransac_thres)

    if H_pred is None:
        return np.nan
    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    return corner_dist

