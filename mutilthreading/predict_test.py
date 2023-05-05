import sys
import torch
import numpy as np
import importlib
import time
import threading

class THREAD(threading.Thread):
    def __init__(self, func, args) :
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(self.args)

    def getResult(self):
        return self.result

# def predict(model_name, x):
#     model = importlib.import_module('rotation')
#     model = model.get_model(normal_channel=False)
#     pred = model(x)

model = importlib.import_module('rotation')
rotation = model.get_model(normal_channel=False)
rotation = rotation.cuda()

model = importlib.import_module('sign')
sign = model.get_model(normal_channel=False)
sign = sign.cuda()

model = importlib.import_module('translation')
translation = model.get_model(normal_channel=False)
translation = translation.cuda()

model = importlib.import_module('width')
width = model.get_model(normal_channel=False)
width = width.cuda()

x = torch.randn(3,8,1024*10)
x = x.cuda()

threads = list()
threads.append(THREAD(rotation, x))
threads.append(THREAD(sign, x))
threads.append(THREAD(translation, x))
threads.append(THREAD(width, x))


start = time.time()

# s = sign(x)
# w = width(x)
# t = translation(x)
# t = rotation(x)

for i in threads:
    i.start()

for i in threads:
    i.join()
    print(i.getResult())


end = time.time()
print("The time used to calculated is ", end - start, 's ')