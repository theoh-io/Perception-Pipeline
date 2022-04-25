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


class ReID_Tracker():
    def __init__(self):
        self.ReID_model=ResNet50(10)
        self.transformation=transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dist_thresh= 8
        self.cos_thresh=0.85
        self.ref_emb=torch.tensor([[]])

    def load_pretrained(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
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

    def distance(self, emb1, emb2, metric):
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

    def image_preprocessing(self, img):
        #Preprocessing: resize so that it fits with the entry of the neural net, convert to tensor type, normalization
        return self.transformation(img)

    def embedding_comparator(self, detections, metric='L2'):
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
                print("error: trying to initialize with multiple detections")
                return None
                #try to handle the case of multiple detections by using the conf_list as input ??
                #for idx in range(detections.size[0]):
                #ref_emb=embedding_generator(ReID_model, detections)
        else:
            #compute L2 distance for each detection

            emb_list=self.embedding_generator(detections)
            #fill a list of all distances wrt to ref
            #dist_list=[]
            dist_list=self.distance(emb_list, torch.unsqueeze(self.ref_emb,0), metric)# .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            if metric=='L2':
                best_dist = min(dist_list)
            elif metric=='cosine':
                best_dist= max(dist_list) #for cosine gives a similarity score between 0 and 1

            best_dist=best_dist.squeeze()
            best_dist=float(best_dist)
            print("best distance between embeddings", best_dist)
            idx=int((dist_list==best_dist).nonzero().squeeze())
            
            #compare to the defined threshold to decide if it's similar enough

            if (metric=='L2' and best_dist < self.dist_thresh):
                #self.ref_emb=emb_list[idx]
                return idx
            elif (metric=='cosine' and best_dist>self.cos_thresh):
                self.ref_emb=emb_list[idx]
                return idx
            else:
                print("under thresh")
                return None
    def average_ref(self):
        #return an average of the reference embeddings
        N=self.ref_emb.shape[0]
        weights=torch.ones((1, N))/N
        if weights.shape[1] != N:
            print("error in size of weights")
        result = torch.matmul(weights, self.ref_emb)
        #print("self_emb shape: ", self.ref_emb.shape)
        #print("results shape :", result.shape)
        return result
        
    def embedding_comparator_mult(self, detections, metric='L2'):
        #update the ref embedding and return the index of correct detection
        idx=None
        #new_ref=None
        best_dist=None
        max_ref=100
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
                print("error: trying to initialize with multiple detections")
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

            dist_list=self.distance(emb_list, average_ref, metric)# self.ref_emb    .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            if metric=='L2':
                best_dist = min(dist_list)
            elif metric=='cosine':
                best_dist= max(dist_list) #for cosine gives a similarity score between 0 and 1

            best_dist=best_dist.squeeze()
            best_dist=float(best_dist)
            print("best distance between embeddings", best_dist)
            idx=int((dist_list==best_dist).nonzero().squeeze())
            
            #compare to the defined threshold to decide if it's similar enough

            if (metric=='L2' and best_dist < self.dist_thresh):
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
            elif (metric=='cosine' and best_dist>self.cos_thresh):
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
                print("under thresh")
                return None