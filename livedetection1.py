import load
import numpy as np
import time
import cv2
import sys
#import threading
#from LoadModel import LoadModel

#LM=LoadModel()
#LM.load()
#t1 = threading.Thread(target=LM.load)
#t1.start()
#print("loaded model {}".format(LM.tfnet))
#print(LM.tfnet)
#t1.join()

#id=0

id=sys.argv[1]
file_name=sys.argv[2]

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(int(id))
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = load.tfnet.return_predict(frame)
        #results = LM.tfnet.return_predict(frame)

        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            print(label)
			#text = '{}: {:.0f}%'.format(label, confidence * 100)
            #frame = cv2.rectangle(frame, tl, br, color, 5)
            #frame = cv2.putText(
            #    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
