
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np

from dlav22.utils.utils import Utils

from dlav22.trackers.kalman_filter import KalmanFilter
from dlav22.trackers.track import Track
from dlav22.deep_sort.deep_sort import DeepSort


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
    def __init__(self, path, metric, dist_thresh, ref_method='multiple', nb_ref=20, av_method='standard', intra_dist=5, verbose=False):
        self.ReID_model=ResNet50(10)
        self.transformation=transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.path=path
        self.metric=metric
        self.dist_thresh=dist_thresh
        self.ref_method=ref_method
        self.nb_ref=nb_ref
        self.av_method=av_method
        self.intra_dist=intra_dist
        self.verbose=verbose
        #self.dist_thresh= 8
        #self.cos_thresh=0.85
        self.ref_emb=torch.tensor([[]])
        self.load_pretrained()

    def load_pretrained(self):
        path=self.path
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


    def track(self, detections):
        ref_method=self.ref_method
        metric=self.metric
        if (ref_method=='multiple'):
            idx=self.embedding_comparator_mult(detections)
        elif (ref_method=='smart'):
            idx=self.embedding_comparator_smart(detections)
        else:
            idx=self.embedding_comparator(detections)
        return idx


class Custom_ReID_with_Deepsort():

    def __init__(self, ds_tracker: DeepSort, reid_tracker: ReID_Tracker) -> None:
        self.ds_tracker = ds_tracker
        self.reid_tracker = reid_tracker
        self.current_ds_track_idx = None
        
    def first_tracking(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        idx_=self.reid_tracker.track(cut_imgs)
        bbox = None
        if idx_ is not None:
            #cut_img = cut_imgs[idx_]
            bbox = self.update_deepsort(detections, idx_, img)
        return bbox

    def track(self, cut_imgs: list, detections: list, img: np.ndarray) -> list:
        '''
        cut_imgs: img parts cut from img at bbox positions
        detections: bboxes from YOLO detector
        img: original image
        -> bbox
        '''
        self.track_with_deepsort(detections,img)
        track_ids = [track.track_id for track in self.ds_tracker.tracker.tracks]
        print('IDs',track_ids)
        print('ID',self.current_ds_track_idx)
        if self.current_ds_track_idx in track_ids:
            idx_ = track_ids.index(self.current_ds_track_idx)
        else:
            idx_ = self.reid_tracker.track(cut_imgs)
            bbox = self.update_deepsort(detections, idx_, img)
        bbox = detections[idx_]
        return bbox

    def update_deepsort(self,detections, idx_, img):
        print("Updating...")
        bbox=detections[idx_]
        self.track_with_deepsort([bbox], img)
        self.current_ds_track_idx = self.ds_tracker.tracker.tracks[0].track_id
        return bbox

    def track_with_deepsort(self, bboxes: list, img: np.ndarray):
        confs = list(np.zeros(len(bboxes)))
        classes = list(np.zeros(len(bboxes)))
        # print(bboxes) #[array([312, 276, 515, 363])]
        for i, bbox_ in enumerate(bboxes):
            bbox_ = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(bbox_)
            bbox_ = Utils.get_xyah_from_tlwh(bbox_)
            bbox_ = [int(b) for b in bbox_] #FIXME more efficient with numpy
            bboxes[i] = bbox_
        # print(bboxes)
        #bboxes = np.array(bboxes)
        bboxes = torch.tensor(bboxes)
        confs = torch.tensor(confs)
        classes = torch.tensor(classes)
        # img = torch.tensor(img)
        self.ds_tracker.update(bboxes, confs, classes, img)
    

class ReID_with_KF():

    def __init__(self, ReID: ReID_Tracker) -> None:
        self.reid_tracker = ReID
        self.kf = KalmanFilter()
        # FIXME Setting up the KF needs a lot of efforts... (to have it clean)
        # Simple KF is not sufficient since one has to include functionality when to stop tracking

    def initiate_track(self, detection):
        # Execite after first detection and when the obj enters the state again
        detection = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(detection)
        detection = Utils.get_xyah_from_tlwh(detection)
        mean, covariance = self.kf.initiate(detection)
        conf = 1.0
        n_init = 0
        max_age = 1000
        feature = None
        self.track = Track(mean, covariance, 0, 0, conf, n_init, max_age, feature)

    def track(self, bboxes: list, img_detection: np.ndarray):
        
        idx_ = self.reid_tracker.track(img_detection)
        bbox_detect = bboxes[idx_]
        self.predict()
        self.update(bbox_detect) #confidences
    
    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    def update(self,detection):
        detection = Utils.get_bbox_tlwh_from_xcent_ycent_w_h(detection)
        detection = Utils.get_xyah_from_tlwh(detection)
        # self.conf = conf
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, detection)
        self.time_since_update = 0



# #################
# #Deepsort Tracker
# #################

# import gdown #needing a quick pip install
# from os.path import exists as file_exists, join

# from .deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
# from .deep_sort.sort.detection import Detection
# from .deep_sort.sort.tracker import Tracker
# from .deep_sort.deep.reid_model_factory import show_downloadeable_models, get_model_link, is_model_in_factory, \
#     is_model_type_in_model_path, get_model_type, show_supported_models
# from .deep_sort.deep.reid.torchreid.utils import FeatureExtractor
# from .deep_sort.deep.reid.torchreid.utils.tools import download_url
# #import needed for tracker.py in sort
# from .deep_sort.sort import kalman_filter
# from .deep_sort.sort import linear_assignment
# from .deep_sort.sort import iou_matching
# from .deep_sort.sort.track import Track
# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort

# class DeepSort(object):
#     #def __init__(self, model, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
#     def __init__(self, deep_sort_model, device, verbose):
#         #modifications: added attributes from tracker in sort

#         #might need to add image preprocessing as in ReID

#         # initialize deepsort from track.py
#         cfg = get_config()
#         cfg.merge_from_file("deep_sort/configs/deep_sort.yaml") #can add an option to pass a specific config
        
#         model=deep_sort_model
#         max_dist=cfg.DEEPSORT.MAX_DIST,
#         max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#         max_age=cfg.DEEPSORT.MAX_AGE
#         n_init=cfg.DEEPSORT.N_INIT
#         nn_budget=cfg.DEEPSORT.NN_BUDGET

#         # models trained on: market1501, dukemtmcreid and msmt17
#         if is_model_in_factory(model):
#             # download the model
#             model_path = join('deep_sort/deep/checkpoint', model + '.pth')
#             if not file_exists(model_path):
#                 gdown.download(get_model_link(model), model_path, quiet=False)

#             self.extractor = FeatureExtractor(
#                 # get rid of dataset information DeepSort model name
#                 model_name=model.rsplit('_', 1)[:-1][0],
#                 model_path=model_path,
#                 device=str(device)
#             )
#         else:
#             if is_model_type_in_model_path(model):
#                 model_name = get_model_type(model)
#                 self.extractor = FeatureExtractor(
#                     model_name=model_name,
#                     model_path=model,
#                     device=str(device)
#                 )
#             else:
#                 print('Cannot infere model name from provided DeepSort path, should be one of the following:')
#                 show_supported_models()
#                 exit()

#         self.max_dist = max_dist
#         metric = NearestNeighborDistanceMetric(
#             "cosine", self.max_dist, nn_budget)
#         #self.tracker = Tracker(   already imported everything here to merge
#             #metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        
#         #attributes from tracker in sort
#         self.metric = metric
#         self.max_iou_distance = max_iou_distance
#         self.max_age = max_age
#         self.n_init = n_init
#         self._lambda = 0

#         self.kf = kalman_filter.KalmanFilter()
#         self.tracks = []
#         self._next_id = 1

#     def track(self,detections, original_image):
#         #replacement of the original track.py file

#         # Process detections
#         for idx, det in enumerate(detections):  # detections per image
#             seen += 1
#             if det is not None and len(det):
#                 output=self.update(det, original_image)

#         return idx


#     def update(self, bbox_xywh, ori_img):
#         #modification: no more conf andc lass as argument
#         #detections was a class 
#         #no need for NMS here
#         #took the entire update from tracker in sort
#         #format problem because of the new detection => modification of match
#         #deleted confs and class from call of self.tracks.update
#         #here only one detection already pruned delete index of detections ??
#         self.height, self.width = ori_img.shape[:2]
#         # generate detections
#         features = self._get_features(bbox_xywh, ori_img)
#         bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
#         #detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)]
#         detections=np.asarray(bbox_tlwh, dtype=np.float)
#         # run on non-maximum supression
#         #boxes = np.array([d.tlwh for d in detections])
#         #scores = np.array([d.confidence for d in detections])

#         # update tracker
#         self.tracker.predict()  #empty the first time
#         ###replacing update function to not take class and conf
#         #self.tracker.update(detections)   #(detections, classes, confidences)

#         # Run matching cascade.
#         matches, unmatched_tracks, unmatched_detections = \
#             self._match(detections)

#         # Update track set.
#         for track_idx, detection_idx in matches:
#             self.tracks[track_idx].update(
#                 self.kf, detections[detection_idx])
#         for track_idx in unmatched_tracks:
#             self.tracks[track_idx].mark_missed()
#         for detection_idx in unmatched_detections:
#             self._initiate_track(detections[detection_idx])
#         self.tracks = [t for t in self.tracks if not t.is_deleted()]

#         # Update distance metric.
#         active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
#         features, targets = [], []
#         for track in self.tracks:
#             if not track.is_confirmed():
#                 continue
#             features += track.features
#             targets += [track.track_id for _ in track.features]
#             track.features = []
#         self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
#         ###end update

#         # output bbox identities
#         outputs = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue

#             box = track.to_tlwh()
#             x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
#             track_id = track.track_id
#             class_id = track.class_id
#             conf = track.conf
#             outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
#         if len(outputs) > 0:
#             outputs = np.stack(outputs, axis=0)
#         return outputs

#     def _match(self, detections):
#         #modification linear assignement formet wrt to detection
#         #

#         # Split track set into confirmed and unconfirmed tracks.
#         confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
#         unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

#         # Associate confirmed tracks using appearance features.
#         matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
#             self._full_cost_metric,
#             linear_assignment.INFTY_COST - 1,  # no need for self.metric.matching_threshold here,
#             self.max_age,
#             self.tracks,
#             detections,
#             confirmed_tracks,
#         )

#         # Associate remaining tracks together with unconfirmed tracks using IOU.
#         iou_track_candidates = unconfirmed_tracks + [
#             k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
#         ]
#         unmatched_tracks_a = [
#             k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
#         ]
#         matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
#             iou_matching.iou_cost,
#             self.max_iou_distance,
#             self.tracks,
#             detections,
#             iou_track_candidates,
#             unmatched_detections,
#         )

#         matches = matches_a + matches_b
#         unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
#         return matches, unmatched_tracks, unmatched_detections

#     def _initiate_track(self, detection, class_id, conf):
#         mean, covariance = self.kf.initiate(detection.to_xyah())
#         self.tracks.append(Track(
#             mean, covariance, self._next_id, class_id, conf, self.n_init, self.max_age,
#             detection.feature))
#         self._next_id += 1
    
#     def predict(self):
#         """Propagate track state distributions one time step forward.

#         This function should be called once every time step, before `update`.
#         """
#         for track in self.tracks:
#             track.predict(self.kf)

#     def increment_ages(self):
#         for track in self.tracks:
#             track.increment_age()
#             track.mark_missed()

#     def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
#         """
#         This implements the full lambda-based cost-metric. However, in doing so, it disregards
#         the possibility to gate the position only which is provided by
#         linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
#         Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
#         distance being normalised, we employ a quick and dirty normalisation based on the
#         threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
#         that the valid values range 0-1.
#         Note also that the authors work with the squared distance. I also sqrt this, so that it
#         is more intuitive in terms of values.
#         """
#         # Compute First the Position-based Cost Matrix
#         pos_cost = np.empty([len(track_indices), len(detection_indices)])
#         msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
#         for row, track_idx in enumerate(track_indices):
#             pos_cost[row, :] = np.sqrt(
#                 self.kf.gating_distance(
#                     tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False
#                 )
#             ) / self.GATING_THRESHOLD
#         pos_gate = pos_cost > 1.0
#         # Now Compute the Appearance-based Cost Matrix
#         app_cost = self.metric.distance(
#             np.array([dets[i].feature for i in detection_indices]),
#             np.array([tracks[i].track_id for i in track_indices]),
#         )
#         app_gate = app_cost > self.metric.matching_threshold
#         # Now combine and threshold
#         cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
#         cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
#         # Return Matrix
#         return cost_matrix
    
#     """
#     TODO:
#         Convert bbox from xc_yc_w_h to xtl_ytl_w_h
#     Thanks JieChen91@github.com for reporting this bug!
#     """
#     @staticmethod
#     def _xywh_to_tlwh(bbox_xywh):
#         if isinstance(bbox_xywh, np.ndarray):
#             bbox_tlwh = bbox_xywh.copy()
#         elif isinstance(bbox_xywh, torch.Tensor):
#             bbox_tlwh = bbox_xywh.clone()
#         bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
#         bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
#         return bbox_tlwh

#     def _xywh_to_xyxy(self, bbox_xywh):
#         x, y, w, h = bbox_xywh
#         x1 = max(int(x - w / 2), 0)
#         x2 = min(int(x + w / 2), self.width - 1)
#         y1 = max(int(y - h / 2), 0)
#         y2 = min(int(y + h / 2), self.height - 1)
#         return x1, y1, x2, y2

#     def _tlwh_to_xyxy(self, bbox_tlwh):
#         """
#         TODO:
#             Convert bbox from xtl_ytl_w_h to xc_yc_w_h
#         Thanks JieChen91@github.com for reporting this bug!
#         """
#         x, y, w, h = bbox_tlwh
#         x1 = max(int(x), 0)
#         x2 = min(int(x+w), self.width - 1)
#         y1 = max(int(y), 0)
#         y2 = min(int(y+h), self.height - 1)
#         return x1, y1, x2, y2

#     def increment_ages(self):
#         self.tracker.increment_ages()

#     def _xyxy_to_tlwh(self, bbox_xyxy):
#         x1, y1, x2, y2 = bbox_xyxy

#         t = x1
#         l = y1
#         w = int(x2 - x1)
#         h = int(y2 - y1)
#         return t, l, w, h

#     def _get_features(self, bbox_xywh, ori_img):
#         im_crops = []
#         for box in bbox_xywh:
#             x1, y1, x2, y2 = self._xywh_to_xyxy(box)
#             im = ori_img[y1:y2, x1:x2]
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops)
#         else:
#             features = np.array([])
#         return features

    
