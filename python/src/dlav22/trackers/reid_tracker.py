import os

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models
from torchvision import transforms

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.cam = False

    def forward(self, x):
        x = self.base(x)
        if self.cam:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ReIdTracker():
    #def __init__(self, path, metric, dist_thresh, ref_method='multiple', nb_ref=20, av_method='standard', intra_dist=5, verbose=False):
    def __init__(self, cfg, path=None, metric=None, dist_thresh=None, ref_method='multiple', nb_ref=20, av_method='standard', intra_dist=5, verbose=False):
        self.ReID_model=ResNet50(10)
        self.transformation=transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        file_path = os.path.dirname(os.path.realpath(__file__))
        path = file_path+cfg.REID.MODEL_PATH
        metric = cfg.REID.SIMILARITY_MEASURE
        dist_thresh = cfg.REID.SIMILARITY_THRESHOLD
        verbose = cfg.PERCEPTION.VERBOSE

        self.path=path
        self.metric=metric
        self.dist_thresh=dist_thresh
        self.ref_method=ref_method
        self.nb_ref=nb_ref
        self.av_method=av_method
        self.intra_dist=intra_dist
        self.verbose=verbose
        self.ref_emb=torch.tensor([[]])
        self.load_pretrained()

    def load_pretrained(self):
        path=self.path
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')
        checkpoint = torch.load(path, map_location=torch.device(device))
        pretrain_dict = checkpoint['state_dict']
        ReID_model_dict = self.ReID_model.state_dict()
        #define a dictionary, k=key, v=values, defined using :
        #drop layers that don't match in size
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in ReID_model_dict and ReID_model_dict[k].size() == v.size()}
        ReID_model_dict.update(pretrain_dict)
        self.ReID_model.load_state_dict(ReID_model_dict)

    def embedding_generator(self, tensor_img):
        #tensor_img=tensor_img.unsqueeze(0) #network expect batch of 64 images
        self.ReID_model.eval()
        #feeds it to the model
        with torch.no_grad():
            embedding =self.ReID_model(tensor_img)
        return embedding

    def distance(self, emb1, emb2):
        metric=self.metric
        #L2 distance
        if metric=='L2':
            if emb1.shape[0]==1:
                return torch.cdist(emb1, emb2, p=2) #.unsqueeze(0)
            else:
                return torch.cdist(emb1, emb2, p=2).squeeze(1)
            
        elif metric=='cosine':
        #cosine similarity
            cos_dist=nn.CosineSimilarity(dim=1, eps=1e-6)
            return torch.unsqueeze(cos_dist(emb1, emb2),0)
        else:
            print("Metric option doesn't exist")

    def image_preprocessing(self, img):
        #Preprocessing: resize so that it fits with the entry of the neural net, convert to tensor type, normalization
        return self.transformation(img)

    def embedding_comparator(self, detections):
        dist_thresh=self.dist_thresh
        metric=self.metric
        #update the ref embedding and return the index of correct detection
        idx=None
        #new_ref=None
        best_dist=None
        #ref_emb is a tensor, detections must be a list of tensor image cropped, conf list array
        if self.ref_emb.nelement() == 0:
            if(detections.size(0)==1):
            #may need to squeeze(0)
                ref_emb=self.embedding_generator(detections)
                idx=0
                self.ref_emb=torch.squeeze(ref_emb)
                return idx
            else:
                if self.verbose is True: print("error: trying to initialize with multiple detections")
                return None
                #try to handle the case of multiple detections by using the conf_list as input ??
                #for idx in range(detections.size[0]):
                #ref_emb=embedding_generator(ReID_model, detections)
        else:
            #compute L2 distance for each detection

            emb_list=self.embedding_generator(detections)
            #fill a list of all distances wrt to ref
            #dist_list=[]
            dist_list=self.distance(emb_list, torch.unsqueeze(self.ref_emb,0))# .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            if metric=='L2':
                best_dist = min(dist_list)
            elif metric=='cosine':
                best_dist= max(dist_list) #for cosine gives a similarity score between 0 and 1

            best_dist=best_dist.squeeze()
            best_dist=float(best_dist)
            if self.verbose is True: print("best distance to average ref", best_dist)
            idx=int((dist_list==best_dist).nonzero().squeeze())
            
            #compare to the defined threshold to decide if it's similar enough

            if (metric=='L2' and best_dist < dist_thresh):
                #self.ref_emb=emb_list[idx]
                return idx
            elif (metric=='cosine' and best_dist>dist_thresh):
                self.ref_emb=emb_list[idx]
                return idx
            else:
                if self.verbose is True: print("under thresh")
                return None

    def average_ref(self):
        #return an average of the reference embeddings
        method=self.av_method
        N=self.ref_emb.shape[0]
        #always use the standard averaging before the ref list is full
        if(N<self.nb_ref):
            method='standard'
        if(method=='standard'):
            weights=torch.ones((1, N))/N
        elif(method=='exp'):
            weights=torch.linspace(1, 10, N)
            weights=torch.exp(weights)
            sum=torch.sum(weights)
            weights=torch.unsqueeze(weights/sum, 0)
        elif(method=='linear'): #standard averaging
            weights=torch.linspace(0.1, 1, N)
            sum=torch.sum(weights)
            weights=torch.unsqueeze(weights/sum, 0)
            #weights=weights.type(torch.DoubleTensor)
        else:
            print("error in method of averaging")
        if weights.shape[1] != N:
            print("error in size of weights")
        result = torch.matmul(weights, self.ref_emb)
        #print("self_emb shape: ", self.ref_emb.shape)
        #print("results shape :", result.shape)
        return result
        
    def embedding_comparator_mult(self, detections):
        metric=self.metric
        dist_thresh=self.dist_thresh
        #update the ref embedding and return the index of correct detection
        idx=None
        #new_ref=None
        best_dist=None
        max_ref=self.nb_ref
        #ref_emb is a tensor, detections must be a list of tensor image cropped, conf list array

        #######################################
        # Initialization with first detection #
        #######################################

        if self.ref_emb.nelement() == 0:
            if(detections.size(0)==1):
            #may need to squeeze(0)
                ref_emb=self.embedding_generator(detections)
                idx=0
                self.ref_emb=ref_emb #mightnot need to cat and just equal
                return idx
            else:
                if self.verbose is True: print("error: trying to initialize with multiple detections")
                return None
                #try to handle the case of multiple detections by using the conf_list as input ??
                #for idx in range(detections.size[0]):
                #ref_emb=embedding_generator(ReID_model, detections)

        else: #compute distance for each detection
            emb_list=self.embedding_generator(detections)
            #fill a list of all distances wrt to ref
            #dist_list=[]
            #average the ref_embeddings
            average_ref=self.average_ref()
            # print("shape of refs: ", self.ref_emb.shape)
            # print("shape of avg: ", average_ref.shape)

            dist_list=self.distance(emb_list, average_ref)# self.ref_emb    .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            if metric=='L2':
                best_dist = min(dist_list)
            elif metric=='cosine':
                best_dist= max(dist_list) #for cosine gives a similarity score between 0 and 1

            best_dist=best_dist.squeeze()
            best_dist=float(best_dist)
            if self.verbose is True: print("best distance to average ref", best_dist)
            idx=int((dist_list==best_dist).nonzero().squeeze())
            
            #compare to the defined threshold to decide if it's similar enough

            if (metric=='L2' and best_dist < dist_thresh):
                #self.ref_emb=emb_list[idx]
                #print("emb_list[idx] shape:", emb_list[idx].shape)
                #print("self.ref_emb shape:", self.ref_emb.shape)
                if self.ref_emb.shape[0]<max_ref:
                    self.ref_emb=torch.cat((self.ref_emb, torch.unsqueeze(emb_list[idx],0)), 0)
                elif self.ref_emb.shape[0]==max_ref:
                    self.ref_emb=torch.cat((self.ref_emb[1:],torch.unsqueeze(emb_list[idx], 0)), 0)
                else :
                    print("error number of ref")
                return idx
            elif (metric=='cosine' and best_dist> dist_thresh):
                #self.ref_emb=emb_list[idx]
                #print("emb_list[idx] shape:", emb_list[idx].shape)
                #print("self.ref_emb shape:", self.ref_emb.shape)
                if self.ref_emb.shape[0]<max_ref:
                    self.ref_emb=torch.cat((self.ref_emb, torch.unsqueeze(emb_list[idx],0)), 0)
                elif self.ref_emb.shape[0]==max_ref:
                    self.ref_emb=torch.cat((self.ref_emb[1:],torch.unsqueeze(emb_list[idx], 0)), 0)
                else :
                    print("error number of ref")
                return idx
            else:
                if self.verbose is True: print("under thresh")
                return None

    def smart_emb_dist(self, new_emb, smart_thresh):
        #if self.verbose is True: print("!!! in smart emb dist")
        ref_list=self.ref_emb
        #if self.verbose is True: print("number of ref", ref_list.shape[0])
        #compute distance to each of the ref
        if ref_list.shape[0]==1:
            intra_dist=torch.cdist(ref_list, new_emb, p=2) #.unsqueeze(0)
        else:
            intra_dist=torch.cdist(ref_list, new_emb, p=2).squeeze(1)

        if self.verbose is True: print("intra dist: ", intra_dist)
        high_dist=max(intra_dist)
        if self.verbose is True: print("highest dist", high_dist)
        if(high_dist>=smart_thresh):
            low_idx=np.argmin(intra_dist)
            return True, low_idx
        return False, None

    def embedding_comparator_smart(self, detections):
        metric=self.metric
        dist_thresh=self.dist_thresh
        smart_thresh=self.intra_dist
        #update the ref embedding and return the index of correct detection
        idx=None
        #new_ref=None
        best_dist=None
        max_ref=self.nb_ref
        #######################################
        # Initialization with first detection #
        #######################################

        if self.ref_emb.nelement() == 0:
            if(detections.size(0)==1):
            #may need to squeeze(0)
                ref_emb=self.embedding_generator(detections)
                idx=0
                self.ref_emb=ref_emb #mightnot need to cat and just equal
                return idx
            else:
                if self.verbose is True: print("trying to initialize with multiple detections")
                return None
                #try to handle the case of multiple detections by using the conf_list as input ??
                #for idx in range(detections.size[0]):
                #ref_emb=embedding_generator(ReID_model, detections)

        else:
            #compute L2 distance for each detection
            emb_list=self.embedding_generator(detections)
            #fill a list of all distances wrt to ref
            #dist_list=[]
            #average the ref_embeddings
            average_ref=self.average_ref()
            # print("shape of refs: ", self.ref_emb.shape)
            # print("shape of avg: ", average_ref.shape)

            dist_list=self.distance(emb_list, average_ref)# self.ref_emb    .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            if metric=='L2':
                best_dist = min(dist_list)
                best_dist_idx=np.argmin(dist_list)
            elif metric=='cosine':
                best_dist= max(dist_list) #for cosine gives a similarity score between 0 and 1
                best_dist_idx=np.argmax(dist_list)
            best_dist=best_dist.squeeze()
            best_dist=float(best_dist)
            if self.verbose is True: print("best distance to average ref", best_dist)
            idx=int((dist_list==best_dist).nonzero().squeeze())
            
            smart_bool, smart_idx=self.smart_emb_dist(torch.unsqueeze(emb_list[idx], 0), smart_thresh)
            #compare to the defined threshold to decide if it's similar enough
            
            if (metric=='L2' and best_dist < dist_thresh): #add smart emb function output bool in condition
                #self.ref_emb=emb_list[idx]
                #print("emb_list[idx] shape:", emb_list[idx].shape)
                #print("self.ref_emb shape:", self.ref_emb.shape)
                if self.ref_emb.shape[0]<max_ref:
                    self.ref_emb=torch.cat((self.ref_emb, torch.unsqueeze(emb_list[idx],0)), 0)
                elif self.ref_emb.shape[0]==max_ref:
                    self.ref_emb=torch.cat((self.ref_emb[1:],torch.unsqueeze(emb_list[idx], 0)), 0)
                else :
                    print("error number of ref")
                return idx
            elif (metric=='cosine' and ((best_dist> dist_thresh and smart_bool) or self.ref_emb.shape[0]<max_ref)):
                #self.ref_emb=emb_list[idx]
                #print("emb_list[idx] shape:", emb_list[idx].shape)
                #print("self.ref_emb shape:", self.ref_emb.shape)
                if self.verbose is True: print("adding ref method: based on intra dist")
                if self.ref_emb.shape[0]<max_ref:
                    self.ref_emb=torch.cat((self.ref_emb, torch.unsqueeze(emb_list[idx],0)), 0)
                elif self.ref_emb.shape[0]==max_ref:
                    #removing smart idx value
                    ref_emb=torch.cat([self.ref_emb[:smart_idx], self.ref_emb[smart_idx+1:]])
                    self.ref_emb=torch.cat((ref_emb,torch.unsqueeze(emb_list[idx], 0)), 0)
                else :
                    print("error number of ref")
                return idx

            elif (metric=='cosine' and best_dist> dist_thresh and not smart_bool):
                if self.verbose is True: print("adding ref method: FIFO")
                if self.ref_emb.shape[0]<max_ref:
                    self.ref_emb=torch.cat((self.ref_emb, torch.unsqueeze(emb_list[idx],0)), 0)
                elif self.ref_emb.shape[0]==max_ref:
                    self.ref_emb=torch.cat((self.ref_emb[1:],torch.unsqueeze(emb_list[idx], 0)), 0)
                else :
                    print("error number of ref")
                return idx
            else:
                if self.verbose is True: print("under thresh")
                return None


    def track(self, cut_imgs, bbox_list, img=None, return_idx=False):
        '''
        cut_imgs: img parts cut from img at bbox positions and converted into tensors
        bbox_list: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        ref_method=self.ref_method
        metric=self.metric
        if (ref_method=='multiple'):
            idx=self.embedding_comparator_mult(cut_imgs)
        elif (ref_method=='smart'):
            idx=self.embedding_comparator_smart(cut_imgs)
        else:
            idx=self.embedding_comparator(cut_imgs)
        if return_idx:
            return idx
        else:
            if idx is None:
                return None
            else:
                return bbox_list[idx]
