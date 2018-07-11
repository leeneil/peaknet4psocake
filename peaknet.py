import darknet as dn
from darknet_utils import *
import cv2
import os


cfgDefault = "cfg/newpeaksv5-asic.cfg"
weightDefault= "weights/newpeaksv5_6240.weights" 
dataDefault = "cfg/peaks.data"

class Peaknet():
    
    
    def __init__(self, cfgPath=cfgDefault, weightPath=weightDefault, dataPath=dataDefault):
        dn.set_gpu(0)
        self.net = dn.load_net( cfgPath, weightPath, 0 )
        self.meta = dn.load_meta( dataPath )
        self.weightPath = weightPath
        self.cfgPath = cfgPath

    def detectBatch(self, imgs, thresh=0.1, hier_thresh=.5, nms=.45):
        if len(imgs.shape) < 3:
            raise Exception("imgs should be 3D or 4D");
        elif len(imgs.shape) == 3:
            imgs = np.reshape( imgs, [1]+list(imgs.shape) )
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

    def psana2cheetah( self, imgs ):
        imgs = np.reshape( imgs, [388, 185, 4, 8] )
        imgs = np.transpose( imgs, [0, 2, 1, 3] );
        img = np.reshape( imgs, [1552, 1480] )
        return img

    def cheetah2psana( self, results ):
        nPeaks = len(results)
        s = np.zeros( (nPeaks,1) )
        r = np.zeros( (nPeaks,1) )
        c = np.zeros( (nPeaks,1) )
        counter = 0
        for u in range(len(results)):
            x = results[u][1][0]
            y = results[u][1][1]
            quad = floor( x / 388.0 )
            seg = floor( y / 185.0 ) 
            s[counter] = seg + 8 * quad
            r[counter] = results[u][1][1] % 185
            c[counter] = results[u][1][0] % 388
            counter += 1
        return s, r, c

    def train( self, imgs, labels, box_size = 7, tmp_path="tmps" ):
        if os.path.isdir(tmp_path):
            pass
        else:
            os.makedirs(tmp_path)
	n, h, w = imgs.shape
	for u in range(n):
	    cv2.imwrite( os.path.join( tmp_path, str(u).zfill(4)+".png" ), imgs[u,:,:] )
	s, r, c = labels
        print(np.max(s))
        counters = np.zeros( (n,1) )
        for u in range(n):
            txt = open( os.path.join( tmp_path, str(u).zfill(4)+".txt"), 'w')
	for u in range(s.shape[0]):
            seg = int(s[u])
            x = c[u][0]
	    y = r[u][0]
            if x + y < 0.1:
            	break
	    try:
		txt = open( os.path.join( tmp_path, str(seg).zfill(4)+".txt"), 'a')
	        x = 1.0*x / w
	        y = 1.0*y / h
                line = '0 {} {} {} {}\n'.format( x, y, box_size, box_size )
	        counters[int(s[u])] += 1
                txt.write(line)
	    finally:            
	        txt.close()
        cwd = os.getcwd()
        os.system("ls $PWD/"+tmp_path+"/*.png > "+tmp_path+"/train.lst")
        txt = open( os.path.join( tmp_path, "peaks.data"), 'w')
        txt.write("classes= 1\n")
        txt.write("train = " + os.path.join( cwd, tmp_path, "train.lst") + "\n" )
        txt.write("names = /reg/neh/home/liponan/ai/darknet/data/peaks.names\n")
        txt.write("backup = backup")
        txt.close()
        train_log = os.popen("cd /reg/neh/home/liponan/ai/psnet && ./darknet detector train " \
				+ os.path.join( cwd, tmp_path, "peaks.data") + " " \
				+ self.cfgPath + " " + self.weightPath ).read()
        print(train_log)
	return np.zerons( (100,100) )
			
		    
        



        
