import numpy as np
# import matplotlib.pyplot as plt
import os
import struct
import gzip

def download():
    '''
    Download files from url
    '''
    os.makedirs(f"{DownloadDir}", exist_ok=True)

    for filename in filenames.values():

        os.system(f"wget -nc -P {DownloadDir} {url}/{filename}")

    return

def make_raw():

    f = gzip.open(f'{DownloadDir}/{filenames["trimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(60000*28*28)
    trimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(60000, 28, 28)/255.    
    print (trimg.shape, trimg.min(), trimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["teimg"]}', mode='r')
    buf = f.read(16)
    buf = f.read(10000*28*28)
    teimg = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(10000, 28, 28)/255.    
    print (teimg.shape, teimg.min(), teimg.max())

    f = gzip.open(f'{DownloadDir}/{filenames["trlbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(60000*1)
    trlbl = np.frombuffer(buf, dtype=np.uint8).reshape(60000, 1) 
    trlbl = np.eye(10)[trlbl]
    print (trlbl.shape, trlbl.min(), trlbl.max())

    f = gzip.open(f'{DownloadDir}/{filenames["telbl"]}', mode='r')
    buf = f.read(8)
    buf = f.read(10000*1)
    telbl = np.frombuffer(buf, dtype=np.uint8).reshape(10000, 1) 
    telbl = np.eye(10)[telbl]
    print (telbl.shape, telbl.min(), telbl.max())

    os.makedirs(f"{RawDir}", exist_ok=True)
    
    trimg.astype('float32').tofile(f"{RawDir}/mnist_trimg.bin")
    trlbl.astype('float32').tofile(f"{RawDir}/mnist_trlbl.bin")
    teimg.astype('float32').tofile(f"{RawDir}/mnist_teimg.bin")
    telbl.astype('float32').tofile(f"{RawDir}/mnist_telbl.bin")
    
    return

if __name__ == "__main__":

    DownloadDir = "Download"
    RawDir = "Raw"
    
    url = "http://yann.lecun.com/exdb/mnist/"

    filenames = {"trimg": "train-images-idx3-ubyte.gz",
                 "trlbl": "train-labels-idx1-ubyte.gz",
                 "teimg": "t10k-images-idx3-ubyte.gz",
                 "telbl": "t10k-labels-idx1-ubyte.gz"}
    
    download()
    
    make_raw()
