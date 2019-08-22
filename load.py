from darkflow.net.build import TFNet

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.2,
}

tfnet = TFNet(options)
