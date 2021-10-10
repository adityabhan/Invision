import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options={
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshhold': '0.2'
    #    'threshold': 0.2,
}

#initialise model
tfnet = TFNet(options)
#enter video file name
capture=cv2.VideoCapture('video1.mp4')

colors=[tuple(255*np.random.rand(3)) for i in range(5)]

#for color in colors:
#    print(color)
while (capture.isOpened()):
    stime=time.time()
    ret,frame=capture.read()#ret bool
    results=tfnet.return_predict(frame)
    if ret:
        for color,result in zip(colors,results):
             tl=(result['topleft']['x'],result['topleft']['y']) #top left
             br=(result['bottomright']['x'],result['bottomright']['y']) #bottom right
             label=result['label'] #label on image
             frame=cv2.rectangle(frame,tl,br,color,10)
             img=cv2.putText(frame,label,tl,cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),5)
        cv2.imshow('frame',frame)
        print('FPS  {:.1f}'.format(1/(time.time()-stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
