import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import glob
import itertools
import argparse

# Parameters
batch_size = 1024
patchSize = 48
stride = 42
halfPatchsize = int(patchSize/2)
modelPath = "./NetworkModel"

def predict(test,ref, sess, pred, x, x_ref, keep_prob, resultDir ):
    windowStep = patchSize - stride
    if(windowStep <= 0):
        windowStep = patchSize
    currentPath = os.path.dirname(os.path.realpath(__file__))
    
    
    #Directories creation
    if not os.path.exists(resultDir):
        os.makedirs(resultDir, exist_ok=True)

    testFilename = test
    referenceFileName = ref
    
    basename, extension = os.path.splitext(os.path.basename(testFilename))    
    
    print("Distorted: ", testFilename, " Reference: ", referenceFileName)
    print("Prediction...")
    
    #Reading images and allocating space for result image
    refImage = cv2.imread(referenceFileName)
    dstImage = cv2.imread(testFilename)
    origWidth, origHeight, channels = refImage.shape

    tmpRefConcat = np.concatenate((np.flip(refImage, 0), refImage, np.flip(refImage, 0)),0)
    refImage = np.concatenate((np.flip(tmpRefConcat, 1), tmpRefConcat, np.flip(tmpRefConcat, 1)),1)
    
    tmpDstConcat = np.concatenate((np.flip(dstImage, 0), dstImage, np.flip(dstImage, 0)),0)
    dstImage = np.concatenate((np.flip(tmpDstConcat, 1), tmpDstConcat, np.flip(tmpDstConcat, 1)),1)
    
    refImage = refImage[origWidth-patchSize:2*origWidth+patchSize, origHeight-patchSize:2*origHeight+patchSize,:]
    dstImage = dstImage[origWidth-patchSize:2*origWidth+patchSize, origHeight-patchSize:2*origHeight+patchSize,:]
    
    width, height, channels = refImage.shape
    
    overlapNumber = int(patchSize / windowStep)
    aggreagtedImage = np.ones((width, height, overlapNumber*overlapNumber))
    
    refRecords = []
    dstRecords = []
    xIndices = []
    yIndices = []
    
    #Patches preparation
    for i in range(0,width-patchSize,windowStep):
        for j in range(0,height-patchSize,windowStep):
            box = (j, i, j+patchSize, i+patchSize)                        
        
            refPatch = refImage[i:i+patchSize,j:j+patchSize]                
            dstPatch = dstImage[i:i+patchSize,j:j+patchSize]              
            
            refRecords.append(refPatch)
            dstRecords.append(dstPatch)
            
            xIndices.append(i)
            yIndices.append(j)
        
    refTuple = tuple(refRecords)
    dstTuple = tuple(dstRecords)            
    
    times = len(refRecords) // batch_size
    mod = len(refRecords) % batch_size
    for i in range(times + 1):
        start = i*batch_size
        end = (i+1)*batch_size
        r = batch_size
        if(i == times):
            end = i*batch_size + mod
            r = mod
            
        predict = sess.run(pred, feed_dict={x: tuple(dstRecords[start:end]) , x_ref:tuple(refRecords[start:end]), keep_prob: 1.})
    
        for index in range(0, r):
            predictedPatch = cv2.resize(predict[index,:,:,:], (patchSize,patchSize))
            
            xx = xIndices[index+ i* batch_size]
            yy = yIndices[index+ i* batch_size]   
            
            layerIndex = int((xx%patchSize)/windowStep)*overlapNumber + int((yy%patchSize)/windowStep)
            aggreagtedImage[xx:xx+patchSize, yy:yy+patchSize, layerIndex] = predictedPatch
    
    vismap = np.mean(aggreagtedImage, axis=2)
        
    vismap = vismap[patchSize:origWidth+patchSize, patchSize:origHeight+patchSize]
    cv2.imwrite(os.path.join(resultDir,basename + extension), vismap)
    
        
 
def usage():
    print("Usage: python cnn_visibility -t [image1-test.png ...] -r [image1-reference.png...]")
    print("       python cnn_visibility -t pathTest/* -r pathRef/*")
 

def multiplePrediction(testList, refList, parameters):
    resultDir ="vismaps"
    try:
        resultDir = parameters['resultDir']
    except:
        print("Use default results dir")
        pass
    
    #Loading net
    ckpt = tf.train.get_checkpoint_state(modelPath)
    print(ckpt)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    print(saver)
    
    pred = tf.get_collection("pred")[0]
    x = tf.get_collection("x")[0]
    x_ref = tf.get_collection("x_ref")[0]
    keep_prob = tf.get_collection("keep_prob")[0]
    
    sess = tf.Session()
    saver.restore(sess, ckpt.model_checkpoint_path)

    print("------- Model From {:s} -------".format(modelPath))
    for test, ref in zip(testList, refList):
        predict(test,ref, sess, pred, x, x_ref, keep_prob, resultDir)
    
def run(parameters):

    try:
        activeGPU = parameters['activeGPU'][0]
        print(activeGPU)
        os.environ["CUDA_VISIBLE_DEVICES"] = activeGPU
    except:
        print("Use default GPU setup")
        pass

    refList = [glob.glob(el) for el in parameters['referenceList']]
    refList = list(itertools.chain(*refList))
    
    testList = [glob.glob(el) for el in parameters['testList']]
    testList = list(itertools.chain(*testList))
 
    if(len(testList) == 0 or len(refList) == 0):
        print("Number of Tests:", len(testList))
        print("Number of References:", len(refList))
        print("One or both lists contain zero elements.")
    elif(len(testList) == len(refList)):
        for l1,l2 in zip(testList, refList):
            print(l1, "  <----->  ",l2)
        multiplePrediction(testList, refList, parameters)
    else:
        print("Different Number of elements. Test and Reference should be paired.")
        print("Number of Tests:", len(testList))
        print("Number of References:", len(refList))

class readable_dir(argparse.Action):
    def __call__(self,parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

def main():
    parameters = {}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testList', nargs='+')
    parser.add_argument('-r', '--referenceList', nargs='+')    
    parser.add_argument('-g', '--activeGPU', nargs='+')
    parser.add_argument('-d', '--resultDir')
    parser.print_help()
    
    args = parser.parse_args()
    
    for key, value in parser.parse_args()._get_kwargs():
        if value is not None:
            parameters[key] = value
    
    #print(parameters)
    run(parameters)
    
        
if __name__ == "__main__":
     main()
  