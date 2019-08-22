
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

#%config InlineBackend.figure_format ='svg'

print("import complete!")
def photoRec(filePath):
    
    options={
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshhold': '0.3'
        }
    tfnet=TFNet(options)

    img = cv2.imread(filePath, cv2.IMREAD_COLOR)#to get rgb image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#color transfoms
    result = tfnet.return_predict(img)
    for i in range(1,len(result)):
        temp=result[i]['confidence']
        if temp > 0.5:
            tl=(result[i]['topleft']['x'],result[i]['topleft']['y']) #top left
            br=(result[i]['bottomright']['x'],result[i]['bottomright']['y']) #bottom right
            label=result[i]['label'] #label on image
            img=cv2.rectangle(img,tl,br,(0,255,0),20)
            img=cv2.putText(img,label,tl,cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),5)
            print(i)
    plt.imshow(img)
    plt.show()
    cv2.imwrite('11.jpg',img)

if __name__ == '__main__':
    photoRec('friends.jpg')

if __name__ == "__gui__":
    print("working!")
    photoRec(sys.argv[0])
