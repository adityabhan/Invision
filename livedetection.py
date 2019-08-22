
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
import sys
import json
import datetime
import threading
import fnmatch
import winsound



def task_of_thread(flag,result,ff):
    #duration = 1  # seconds
    #freq = 1440  # Hz
    #winsound.Beep(freq, duration)
    winsound.PlaySound('sound1.wav', winsound.SND_FILENAME)
    if(flag['value'] == 'true'):# and os.path.isfile(ff) == 'False'):
        with open(ff,'w') as f:
            json.dump([{"label":result['label'],"time":datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),"threshold":result['confidence']}],f)
            f.close()
        flag['value']='false'

    else:
        jsonFile = open(ff, "r+")
        data = json.load(jsonFile)
        jsonFile.close()
        print(data.append({"label":result['label'],"time":datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),"threshold":result['confidence']}))

        with open(ff,'w') as f:
            json.dump(data,f)
            f.close


options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.5
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

r=int(sys.argv[1])
username=sys.argv[2]
lab=sys.argv[3]

#r=0
#username="atul"

capture = cv2.VideoCapture(r)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

flag={"value":'true'}
#print(id(flag))
dir=os.getcwd()
ee=datetime.datetime.now().strftime('%Y-%m-%d')#_%H:%M:%S')
pa=os.path.join(dir,'flaskblog\\logs\\',username+'_'+ee+'.json')
#rr=open(pa,'w')
#flag='true
print(pa)
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence=result['confidence']
            result['confidence'] = str(result['confidence'])
            if(label==lab):
                t1=threading.Thread(target=task_of_thread,args=(flag,result,pa, ))
                t1.start()
                t1.join()
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
