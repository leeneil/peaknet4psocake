import darknet as dn
from darknet_utils import *


cfgDefault = "/reg/neh/home/liponan/ai/psnet/cfg/newpeaksv5-test.cfg"
weightDefault= "/reg/neh/home/liponan/ai/psnet/backup/newpeaksv5_6240.weights" 
# cfgDefault = "/reg/neh/home/liponan/ai/darknet/cfg/-peaks.cfg"
# weightDefault= "/reg/neh/home/liponan/ai/psnet/backup/yolov3-peaks_6240.weights" 
dataDefault = "/reg/neh/home/liponan/ai/darknet/cfg/peaks.data"

class peaknet():
    
    
    def __init__(self, cfgPath=cfgDefault, weightPath=weightDefault, dataPath=dataDefault):
        dn.set_gpu(0)
        self.net = dn.load_net( cfgPath, weightPath, 0 )
        self.meta = dn.load_meta( dataPath )

    def detectBatch(self, imgs, thresh=0.1, hier_thresh=.5, nms=.45):
        if len(imgs.shape) < 3:
            raise Exception("imgs should be 3D or 4D");
        elif len(imgs.shape) == 3:
            imgs = np.reshape( imgs, (1)+imgs.shape )
        else:
            pass
        n, m, h, w = imgs.shape
        imgResults = []
        for u in range(n):
            asicResults = []
            for v in range(m):
                #print(imgs[u,v,:,:].shape)
                result = self.detect( imgs[u,v,:,:], thresh=thresh, hier_thresh=hier_thresh, nms=nms)
                asicResults.append( result )
            imgResults.append( asicResults )
        return imgResults
           
    def detect(self, img, thresh=0.1, hier_thresh=.5, nms=.45):
        img = array2image(dn, img)
        boxes = dn.make_boxes(self.net)
        probs = dn.make_probs(self.net)
        num =   dn.num_boxes(self.net)
        dn.network_detect(self.net, img, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            if probs[j][0] > thresh:
                res.append((probs[j][0], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[0])
        dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
        return res

    def peaknet2psana( self, results ):
        nPeaks = 0
        for u in range(len(results)):
            nPeaks += len(results[u])
        s = np.zeros( (nPeaks,1) )
        r = np.zeros( (nPeaks,1) )
        c = np.zeros( (nPeaks,1) )
        counter = 0
        for u in range(len(results)):
            for v in range(len(results[u])):
                s[counter] = u
                r[counter] = results[u][v][1][1]
                c[counter] = results[u][v][1][0]
                counter += 1
            return s, r, c
        
