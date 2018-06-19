# peaknet4psocake
peaknet API

## setup

Add the following to your '~/.bashrc'

```
export PYTHONPATH=/reg/neh/home/liponan/ai/psnet:/reg/neh/home/liponan/ai/psnet/examples:/reg/neh/home/liponan/ai/psnet/python:$PYTHONPATH
```

## usage

First create a peaknet instance, i.e.

```
psnet = peaknet()
```

then call `detect()`, i.e.

```
res = psnet.detect(img, thresh=0.25)
```

where `img` is a 2-D numpy array, `thresh` is the detection threshold in the range [0,1]. The output `res` will be a list of tutples that look like (score, (x,y,w,h)).
