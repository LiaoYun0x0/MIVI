import torch
import torch.nn as nn
from models.resnet.resnet import resnet18,resnet50
from models.transformer import LocalFeatureTransformer,GlobalFeatureTransformer,PositionEncodingSine
import sys
import math

sys.path.append("../")
cfg = {}
cfg["arc_m"] = 0.2
cfg["arc_s"] = 64
cfg["local_feature_dim"] = 256
cfg["global_feature_dim"] = 256
cfg['temp_bug_fix'] = False
cfg["model_weights"] = "./weights/weights_lo_1019/GLNet_55000_481.488.tar"

# global transformer parameters
cfg["gl_cfg"] = {}
gl_cfg = cfg["gl_cfg"]
gl_cfg["d_model"] = 256
gl_cfg["layer_names"] = ["self"] * 6
gl_cfg["nhead"] = 8
gl_cfg["attention"] = "linear"



# local transformer parameters
cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 256
lo_cfg["layer_names"] = ["self","cross"] * 4
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"


def normalize(x):
    '''
    scale the vector length to 1.
    params:
        x: torch tensor, shape "[...,vector_dim]"
    '''
    norm = torch.sqrt(torch.sum(torch.pow(x,2),-1,keepdims=True))
    return x / (norm)

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad, bias=False):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                           padding=(kernel_size - 1) // 2, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        #self.elu = nn.ELU(inplace=True)
        self.mish = nn.Mish(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        #x = self.elu(x)
        x = self.mish(x)
        return x


class GLNet(torch.nn.Module):
    def __init__(self,config=cfg,backbone="resnet18"):
        super(GLNet,self).__init__()
        self.backbone = eval(backbone)(include_top=False)
        
        if backbone == "resnet50":
            self.feature_conv16 = ConvBN(1024,config["local_feature_dim"],1,1,0,bias=False)
            self.feature_conv32 = ConvBN(2048,config["global_feature_dim"],1,1,0,bias=False)
            self.feature_conv8 = ConvBN(512,config["global_feature_dim"],1,1,0,bias=False)
        else: 
            self.feature_conv16 = ConvBN(256,config["local_feature_dim"],1,1,0,bias=False)
            self.feature_conv32 = ConvBN(512,config["global_feature_dim"],1,1,0,bias=False)
        self.pos_encoding = PositionEncodingSine( 
            config['global_feature_dim'], # suppose global_feature_dim equals to local_feature_dim
            temp_bug_fix=config['temp_bug_fix'])
        self.global_transformer = GlobalFeatureTransformer(config["gl_cfg"])
        self.local_transformer = LocalFeatureTransformer(config["lo_cfg"])
        
    
    def attention_global_feature(self,x):
        b,c,h,w = x.shape
        x = self.pos_encoding(x)
        x = x.view(b,c,h*w).transpose(1,2)
        x = self.global_transformer(x)
        x = x.mean(1)
        return x
    
    def attention_local_feature(self,x0,x1):
        b,c,h0,w0 = x0.shape
        _,_,h1,w1 = x1.shape
        x0 = self.pos_encoding(x0)
        x1 = self.pos_encoding(x1)
        x0 = x0.view(b,c,h0*w0).transpose(1,2)
        x1 = x1.view(b,c,h1*w1).transpose(1,2)
        x0,x1 = self.local_transformer(x0,x1)
        return x0,x1

    def forward_pair_lo(self,pbatch,abatch):
        lf0_2x, lf0_4x, lf0_8x, lf0_16x, lf0_32x = self.backbone.extract_endpoints(pbatch)
        lf1_2x, lf1_4x, lf1_8x, lf1_16x, lf1_32x = self.backbone.extract_endpoints(abatch)

        lf0 = self.feature_conv8(lf0_8x) 
        lf1 = self.feature_conv8(lf1_8x)
        lf0,lf1 = self.attention_local_feature(lf0,lf1)

        lf0_16x = self.feature_conv16(lf0_16x) 
        lf1_16x = self.feature_conv16(lf1_16x)
        lf0_16x,lf1_16x = self.attention_local_feature(lf0_16x,lf1_16x)

        return (lf0, lf1), (lf0_2x, lf1_2x), (lf0_16x, lf1_16x), (lf0_4x, lf1_4x),

    def forward_pair(self,pbatch,abatch,cut_lo=False):
        _,_,lf0,gf0 = self.backbone.extract_endpoints(pbatch)
        _,_,lf1,gf1 = self.backbone.extract_endpoints(abatch)


        if cut_lo:
            with torch.no_grad():
                lf0 = self.feature_conv16(lf0) 
                lf1 = self.feature_conv16(lf1)
        else:
            lf0 = self.feature_conv16(lf0) 
            lf1 = self.feature_conv16(lf1)
        lf0,lf1 = self.attention_local_feature(lf0,lf1)

        gf0 = self.feature_conv32(gf0)
        gf1 = self.feature_conv32(gf1)
        gf0 = self.attention_global_feature(gf0)
        gf1 = self.attention_global_feature(gf1)
        return lf0,lf1,normalize(gf0),normalize(gf1)

    def forward(self,batch):
        gl_feature = self.backbone.extract_features(batch)
        global_features = self.feature_conv32(gl_feature)
        global_features = self.attention_global_feature(global_features)
        return normalize(global_features)


class GLMetric():
    def __init__(self,arc_m,arc_s,temperature=0.1,max_batch_size=256):
        self.arc_patch = ArcPatch(m=arc_m,s=arc_s)
        self.temperature = temperature
        

    def CrossEntropyLoss(self,logit,label):
        log_softmax_logit = torch.log(torch.softmax(logit,dim=-1)+1e-8)
        loss = (-label * log_softmax_logit).sum() / logit.shape[0]
        return loss
    
    def compute_gf_loss(self,query_gf,refer_gf,query_bank,refer_bank,neg_bank_mask,use_arc=True):
        batch_size = query_gf.shape[0]
        bank_size = query_bank.shape[0]
        
        qr_label = torch.eye(batch_size)
        sm_inner_batch_qr = torch.matmul(query_gf,refer_gf.T) # (batch_size,batch_size)
        sm_inner_batch_qq = torch.matmul(query_gf,query_gf.T)[qr_label==0].reshape(batch_size,batch_size - 1) # (batch_size, batch_size - 1) 
        neg_query_bank = query_bank[neg_bank_mask] 
        neg_refer_bank = refer_bank[neg_bank_mask]
        sm_q_bq = torch.matmul(query_gf,neg_query_bank.T) # (batch_size, memory - batch_size)
        sm_q_br = torch.matmul(query_gf,neg_refer_bank.T) # (batch_size, memory - batch_size)
        sm = torch.cat([sm_inner_batch_qr,sm_inner_batch_qq,sm_q_bq,sm_q_br],dim=-1) # (batch_size, 2*memory-1)
        label = torch.zeros(batch_size,2*bank_size-1).cuda()
        label[:,:batch_size] = qr_label
        if use_arc:
            sm = self.arc_patch(sm, label > 0)
        else:
            sm = sm / self.temperature
        loss = self.CrossEntropyLoss(sm, label)
        return loss

        
    
    def _compute_confidence_matrix(self,query_lf,refer_lf,gt_matrix=None):
        _d =  query_lf.shape[-1]
        query_lf = query_lf / _d
        refer_lf = refer_lf / _d 
        similarity_matrix = torch.matmul(query_lf,refer_lf.transpose(1,2)) / self.temperature
        confidence_matrix = torch.softmax(similarity_matrix,1) * torch.softmax(similarity_matrix,2)
        return confidence_matrix
    

    def compute_lf_loss(self,lf0,lf1,gt_matrix,use_arc=True):
        """
        lf0: 16x256x400
        lf1: 16x256x400
        gt_m: 400x400
        """
        if use_arc:
            confidence_matrix = self._compute_confidence_matrix(lf0,lf1,gt_matrix=gt_matrix)
        else:
            confidence_matrix = self._compute_confidence_matrix(lf0,lf1)

        loss = (-gt_matrix * torch.log(confidence_matrix + 1e-6)).sum() / lf0.shape[0]
        import pdb
        pdb.set_trace()

        if torch.isnan(loss):
            torch.save(gt_matrix,"gt_matrix.npy")
            torch.save(confidence_matrix,"confidence_matrix.npy")
            torch.save(lf0.shape[0],"shape.npy")
            print("nan_loss occur, stop training")
            sys.exit()
        return loss


if __name__ == "__main__":
    m1 = GLNet()
    
    for p in m1.parameters():
        p.data = p.data * 2 + 1

    for p in m1.parameters():
        print(p)
    

    
        

        
