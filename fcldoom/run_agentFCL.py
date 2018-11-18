from __future__ import print_function
import numpy as np
import cv2
import tensorflow as tf
import sys
import os
sys.path.append('./agent')
sys.path.append("../.")
from agent.doom_simulator import DoomSimulator
import feedback_closedloop_learning
import threading
from matplotlib import pyplot as plt
from datetime import datetime
import random


width = 160
widthIn = 160
height = 120
heightIn = 120
nFiltersInput = 0
nFiltersHidden = 0
nHidden = [5]
nOut = 6
# nFiltersHidden = 0 means that the layer is linear without filters
minT = 5
maxT = 10

outFile = open("/home/paul/Dev/GameAI/vizdoom_cig2017/DFLOutput.txt", "w")
wtdistFile = open("/home/paul/Dev/GameAI/vizdoom_cig2017/wtDist.txt", "w")

FCLNet = feedback_closedloop_learning.FeedbackClosedloopLearning(width * height, nHidden, nOut, nFiltersInput, nFiltersHidden, minT, maxT)
# init the weights
# FCLNet.getLayer(0).setConvolution(width, height)
FCLNet.initWeights(1., feedback_closedloop_learning.Neuron.MAX_OUTPUT_RANDOM)
print ("Initialised weights")
for i in range(len(nHidden)):
    print ("hidden ", i, ": ", nHidden[i], file=outFile)
#print("learning rate: ", learningRate, file=outFile)

FCLNet.setBias(1)
FCLNet.setMomentum(0.5)
random.seed(datetime.now())
FCLNet.seedRandom(np.random.randint(low=0, high=999999))
FCLNet.setUseDerivative(0)

preprocess_input_images = lambda x: x / 255. - 0.5

sharpen = np.array((
	[0, 1, 0],
	[1, 4, 1],
	[0, 1, 0]), dtype="int")

edge = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")


plt.ion()
plt.show()
ln1 = False
ln2 = [False,False,False,False]

def getWeights2D(neuron):
    n_neurons = FCLNet.getLayer(0).getNneurons()
    n_inputs = FCLNet.getLayer(0).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        if FCLNet.getLayer(0).getNeuron(neuron).getMask(i):
            weights[i] = FCLNet.getLayer(0).getNeuron(neuron).getAvgWeight(i)
        else:
            weights[i] = np.nan
    return weights.reshape(heightIn,widthIn)

def getWeights1D(layer,neuron):
    n_neurons = FCLNet.getLayer(layer).getNneurons()
    n_inputs = FCLNet.getLayer(layer).getNeuron(neuron).getNinputs()
    weights = np.zeros(n_inputs)
    for i in range(n_inputs):
        weights[i] = FCLNet.getLayer(layer).getNeuron(neuron).getAvgWeight(i)
    return weights

def plotWeights():
    global ln1
    global ln2

    print("** Update plot")
    while True:

        if ln1:
            ln1.remove()
        plt.figure(1)
        w1 = getWeights2D(0)
        for i in range(1,FCLNet.getLayer(0).getNneurons()):
            w2 = getWeights2D(i)
            w1 = np.where(np.isnan(w2),w1,w2)
        ln1 = plt.imshow(w1,cmap='gray')
        plt.draw()
        plt.pause(0.1)

        for j in range(1,FCLNet.getNumHidLayers()+1):
            if ln2[j]:
                ln2[j].remove()
            plt.figure(j+1)
            w1 = np.zeros( (FCLNet.getLayer(j).getNneurons(),FCLNet.getLayer(j).getNeuron(0).getNinputs()) )
            for i in range(FCLNet.getLayer(j).getNneurons()):
                w1[i,:] = getWeights1D(j,i)
            ln2[j] = plt.imshow(w1,cmap='gray')
            plt.draw()
            plt.pause(5.0)


#t1 = threading.Thread(target=plotWeights)
#t1.start()


def getColourImbalance(img, colour):
    if(img.shape[0]) != 3:
        print("Error in getColourImbalance: wrong number of image channels: ", img.shape)
        return 0.

    width = int(img.shape[2]/2)
    height = int(img.shape[1]/2)
    print ("width: ", width, "height", height)
    avgLeft = np.average(img[:,:,:width], axis=1)
    avgLeft = np.average(avgLeft, axis=1)
#    avgLeft = np.dot(avgLeft, colour)
    avgRight = np.average(img[:,:,width:], axis=1)
    avgRight = np.average(avgRight, axis=1)
#    avgRight = np.dot(avgRight, colour)
    avgTop = np.average(img[:, :height, :], axis=1)
    avgTop = np.average(avgTop, axis=1)
    #    avgTop = np.dot(avgTop, colour)
    avgBottom = np.average(img[:, height:, :], axis=1)
    avgBottom = np.average(avgBottom, axis=1)
    #    avgBottom = np.dot(avgBottom, colour)
    print("avgLeft: ", avgLeft, " avgRight: ", avgRight, "avgTop", avgTop, "avgBottom", avgBottom)
    return 1.

def getMaxColourPos(img, colour, step):
    cv2.imwrite("/home/paul/tmp/Images/Positive/col-" + str(step) + ".jpg", img)

    img = np.array(img, dtype='float64')
    width = int(img.shape[1])
    height = int(img.shape[0])
#    img[:,10,10] = [0,0,255]
    diff = np.ones(img.shape)
    diff[:,:,0] = colour[0]
    diff[:,:,1] = colour[1]
    diff[:,:,2] = colour[2]
    diff = np.absolute(np.add(diff, (-1*img)))
    cv2.imwrite("/home/paul/tmp/Images/Positive/diff-" + ".jpg", diff)
    diff = np.sum(diff, axis=2)

    indx = np.argmin(diff)
    indx0 = int(indx / width)
    indx1 = indx % width
    pts = np.asarray(np.where((np.mean(diff) - diff) > 150))
    if (pts.shape[1]>0):
        bottomLeft = np.array([np.amin(pts[1]), np.amin(pts[0])])
        topRight = np.array([np.amax(pts[1]), np.amax(pts[0])])
    else:
        bottomLeft = []
        topRight = []
    return np.array([indx1, indx0]), bottomLeft, topRight, np.mean(diff) - diff[indx0,indx1]


def savePosImage(curr_step, centre, x1, y1, x2, y2, _img, myFile, width, height):
#    print ("img shape: ", img2.shape)
    myFile.write("/home/paul/tmp/Images/" + str(curr_step) + ".jpg"
                 + " 1"
                 + " " + str(x1) + " " + str(y1)
                 + " " + str(x2) + " " + str(y2) + "\n")
    img = np.zeros(_img.shape,dtype=np.uint8)
    outImage = Image.fromarray(img)
    outImage.save("/home/paul/tmp/Images/Positive/" + str(curr_step) + ".jpg")


def saveImage(curr_step, _img):
    cv2.imwrite("/home/paul/tmp/Images/Positive/DFL-" + str(curr_step) + ".jpg", _img)


def saveNegImage(curr_step, img2, myFile, width, height):
    myFile.write("/home/paul/tmp/Images/" + str(curr_step) + ".jpg\n")
    img = Image.fromarray(img2)
    img.save("/home/paul/tmp/Images/Negative/" + str(curr_step) + ".jpg")


def main(learning_rate_):
    learningRate = float(learning_rate_)
    FCLNet.setLearningRate(learningRate)

    print("learning rate ", learningRate, file=outFile)

    ## Simulator
    simulator_args = {}
    simulator_args['config'] = 'config/config.cfg'
    simulator_args['resolution'] = (widthIn,heightIn)
    simulator_args['frame_skip'] = 1
    simulator_args['color_mode'] = 'RGB24'
    simulator_args['game_args'] = "+name FCL +colorset 7"


    historyLen = 3
    print ("HistoryLen: ", historyLen)

    print('starting simulator')
    simulator = DoomSimulator(simulator_args)
    num_channels = simulator.num_channels

    print('started simulator')

    modelDir = os.path.join(os.path.expanduser("~"), "Dev/GameAI/vizdoom_cig2017/icodoom/ICO1/Models")


    img_buffer = np.zeros(
        (historyLen, simulator.resolution[1], simulator.resolution[0], num_channels), dtype='uint8')

    meas_buffer = np.zeros((historyLen, simulator.num_meas))
    act_buffer = np.zeros((historyLen, 7))
    curr_step = 0
    term = False


    diff_z = 0
    iter = 1
    epoch = 200
    radialFlowLeft = 30.
    radialFlowRight = 30.
    radialFlowInertia = 0.4
    radialGain = 4.
    rotationGain = 50.
    errorThresh = 10.
    updatePtsFreq = 50
    reflexGain = 1E-3
    flowGain = 0.
    netGain = 40.
    reflexReduceGain = -0.05

    # create masks for left and right visual fields - note that these only cover the upper half of the image
    # this is to help prevent the tracking getting confused by the floor pattern
    half_height = round(height/2)
    half_width = round(width/2)

    maskLeft = np.zeros([height, width], np.uint8)
    maskLeft[half_height:, :half_width] = 1.
    maskRight = np.zeros([height, width], np.uint8)
    maskRight[half_height:, half_width:] = 1.

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)

    imgCentre = np.array([int(simulator_args['resolution'][0] / 2), int(simulator_args['resolution'][1] /2)])
    print ("Image centre: ", imgCentre)
    rawInputs = np.zeros((height, width))
    cheatInputs = np.zeros((width, height))
    input_buff = np.zeros((width*height))
    target_buff = np.zeros((1,1))
    meas_buff = np.zeros((1,simulator.num_meas))
    netOut = 0.
    netErr = np.zeros(nHidden[0])
    delta = 0.
    shoot = 0
    wtDist = np.zeros(FCLNet.getNumLayers())

    reflexOn = False
    iter = 0
    killed = False
#    FCLNet.saveModel("Models/hack.txt")

    while not term:
        if curr_step < historyLen:
            curr_act = np.zeros(7).tolist()
            img, meas, rwrd, term = simulator.step(curr_act)
            print("Image: ", img.shape, " max: ", np.amax(img), " min: ", np.amin(img))

            if curr_step == 0:
                p0Left = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskLeft, **feature_params)
                p0Right = cv2.goodFeaturesToTrack(img[:,:,0], mask=maskRight, **feature_params)

            img_buffer[curr_step % historyLen] = img
            meas_buffer[curr_step % historyLen] = meas
            act_buffer[curr_step % historyLen] = curr_act[:7]

        else:
            img1 = img_buffer[(curr_step-2) % historyLen,:,:,:]
            img2 = img_buffer[(curr_step-1) % historyLen,:,:,:]
            state = simulator._game.get_state()

            stateImg = state.screen_buffer

            icoSteer = 0.

            if curr_step > 100:
                health = meas[1]

                if (health<0.1):
                    reflexOn = False
                    iter = 0

                if (simulator._game.is_player_dead()) and killed == False:
                    g = open("/home/paul/Dev/GameAI/vizdoom_cig2017/KD.txt", "a")
                    s = "0 " + str(curr_step) + " " + str(datetime.now().timestamp()) + "\n"

                    g.write(s)
                    g.close()
                    killed = True
                    print("KILLED")
                if (not(simulator._game.is_player_dead())):
                    killed = False

                # Don't run any networks when the player is dead!
                if (health < 101. and health > 0.):

                    icoInSteer = 0.

                    saveImage(curr_step, stateImg)
                    centre, bottomLeft, topRight, colourStrength = getMaxColourPos(stateImg, [255, 0, 0], curr_step)
                    colourSteer = imgCentre[0]

                    if(len(bottomLeft)>0 and len(topRight)>0 and ((topRight[0] - bottomLeft[0]) < width/3) and ((topRight[1] - bottomLeft[1]) < height/2)):
                        colourSteer = bottomLeft[0] + int(0.5 * (topRight[0] - bottomLeft[0]))
                        shoot = 1
#                        cheatInputs = np.copy(stateImg)
#                        cv2.rectangle(cheatInputs, (bottomLeft[0], bottomLeft[1]), (topRight[0], topRight[1]), (0, 0, 255), 2)
#                        cv2.rectangle(cheatInputs, (384, 0), (510, 128), (0, 255, 0), 3)
#                        cv2.imwrite("/home/paul/tmp/Backup/rect-" + str(curr_step) + ".jpg", cheatInputs)

                    rawInputs = np.array(np.sum(stateImg, axis=2) / 3)
#                    cv2.imwrite("/home/paul/tmp/Backup/raw-" + str(curr_step) + ".jpg", rawInputs)

                    input_buff[:] = np.ndarray.flatten(rawInputs)
                    input_buff = input_buff - np.mean(input_buff)
                    input_buff = input_buff / np.sqrt(np.var(input_buff))

                    # we want the reflex to be delayed wrt to the image input, so that the image is. Otherwise the learning can
                    # never reduce the error to zero no matter how good the controller.

                    oldDelta = delta
                    if (iter>2):
                        delta = (float(colourSteer) - float(imgCentre[0]))/float(width)
                    else:
                        delta = 0

                    deltaDiff = delta - oldDelta
                    if(iter>2):
                        if(np.abs(delta) > 0.01):
                            shoot = 0

                    netErr[:] = delta
                    target_buff[...] = delta + netOut
                    meas_buff[0,:] = meas


                    FCLNet.setLearningRate(0.)
                    FCLNet.doStep(input_buff, netErr)
                    netOut = FCLNet.getOutput(0) + 0.3*FCLNet.getOutput(1) + 0.1*FCLNet.getOutput(2)
                    netOut1 = FCLNet.getOutput(3) + 0.3*FCLNet.getOutput(4) + 0.1*FCLNet.getOutput(5)

                    netErr += reflexReduceGain * netGain * (netOut - netOut1)

                    if curr_step > 1790:
                        FCLNet.setLearningRate(learningRate)

                    FCLNet.doStep(input_buff, netErr)
                    netOut = FCLNet.getOutput(0) + 0.3*FCLNet.getOutput(1) + 0.1*FCLNet.getOutput(2)
                    netOut1 = FCLNet.getOutput(3) + 0.3*FCLNet.getOutput(4) + 0.1*FCLNet.getOutput(5)

#                    print("%s" % (" SHOOT " if shoot == 1 else "       "), deltaDiff, delta, netOut)

                    for i in range(FCLNet.getNumLayers()):
                        wtDist[i] = FCLNet.getLayer(i).getWeightDistanceFromInitialWeights()

                    print(curr_step, delta, netErr[0], netOut-netOut1, health, file=outFile)
                    print(' '.join(map(str, wtDist)), file=wtdistFile)

                    diff_theta = 0.6 * max(min((icoInSteer), 5.), -5.)

                    netErr[:] = 0.
                    diff_theta = diff_theta + reflexGain * colourStrength * delta
#                    diff_z = -1.

                    curr_act = np.zeros(7).tolist()
                    curr_act[0] = 0
                    curr_act[1] = 0
                    curr_act[2] = shoot
                    curr_act[3] = curr_act[3] + diff_z
                    curr_act[4] = 0
                    curr_act[5] = 0.
                    curr_act[6] = diff_theta + netGain*(netOut - netOut1)

                    iter += 1

            if (curr_step % epoch == 0):

                if not os.path.exists("Models"):
                    os.makedirs("Models")
#                FCLNet.saveModel("Models/BP-" + str(curr_step) + ".txt")

                file = open("Models/checkpoint", 'w')
                file.write("Models/BP-" + str(curr_step) + ".txt")
                file.close()

            img, meas, rwrd, term = simulator.step(curr_act)
            if (not (meas is None)) and meas[0] > 30.:
                meas[0] = 30.

            if not term:
                img_buffer[curr_step % historyLen] = img
                meas_buffer[curr_step % historyLen] = meas
                act_buffer[curr_step % historyLen] = curr_act[:7]
        curr_step += 1


    simulator.close_game()
    outFile.close()
    wtdistFile.close()

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        print("learning rate: ", str(sys.argv[1]))
        main(sys.argv[1])
    else:
        print("usage: run_agentDFL learning_rate")
