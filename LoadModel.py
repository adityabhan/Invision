from darkflow.net.build import TFNet
import threading

global tfnet
class LoadModel:
    tfnet=None
    def load(self):
        options = {
                  'model': 'cfg/yolo.cfg',
                  'load': 'bin/yolov2.weights',
                  'threshold': 0.2,
                 }
        tfnet = TFNet(options)


if __name__ == '__main__':
    LM=LoadModel()
    #t1 = threading.Thread(target=LM.load)
    #t1.start()
    LM.load()
    print(threading.currentThread().getName())
    print("loaded model {}".format(LM.tfnet))
    print((tfnet))
