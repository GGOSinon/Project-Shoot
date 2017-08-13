import tensorflow as tf
import subprocess
import numpy as np
import sys
import time
import random
learning_rate = 0.01
training_iters = 200000
mem_size = 128

n_input = 128*128
n_classes = 5
dropout = 0.75
#mem_play = [][]
p = 1.0

weights = {
    'wc1': tf.Variable(tf.random_normal([3,3,1,32])),
    'wc2': tf.Variable(tf.random_normal([3,3,32,64])),
    'wc3': tf.Variable(tf.random_normal([3,3,64,64])),
    'wd1': tf.Variable(tf.random_normal([16*16*64, 128])),
    'out': tf.Variable(tf.random_normal([128, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1],padding='SAME')

def norm(x):
    return tf.contrib.slim.batch_norm(x)

def fcl(x, W, b):
    x = tf.reshape(x, [-1,W.get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x,W), b)
    return x;
    
def neural_net(x, weight, biases, dropout):
    x = tf.reshape(x, shape = [-1, 128, 128, 1])
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    #conv1 = norm(conv1)
    #print(conv1.shape) 
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    #conv2 = norm(conv2)
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    #conv3 = norm(conv3)

    fc1 = fcl(conv3, weights['wd1'], biases['bd1'])
    
    out = fcl(fc1, weights['out'], biases['out'])
    return out

step = 0

def print_out(S):
    Msg = open("Msg.txt", "w")
    Msg.write(S+"\n")
    Msg.close()
    print("2X")

def move(x):
    if x==1 : S="forward"
    elif x==2 : S="backward"
    elif x==3 : S="left"
    elif x==4 : S="right"
    elif x==0 : S="shoot"
    else: S="NONE"
    get_signal()
    write_log("Sent "+S+" at step " + str(step))
    print_out(S)

def get_signal():
    line = raw_input()
    while line!="CX" : line+=raw_input()

def get_input():
    get_signal()
    Msg = open("Msg.txt", "r")
    line = Msg.readline().rstrip("\n")
    if len(line)>10 : write_log("Received message length of " + str(len(line)))
    else : write_log("Received message : " + line)
    Msg.close()
    return line
    
def get_image():
    res = []
    #sys.stdout.write("Waiting at step " + str(step))
    line = get_input()
    write_log("Recieved image length of " + str(len(line)))
    for i in range(n_input):
        x = int(line[3*i:3*i+3])
        res.append(np.float32(float(x)))
    return np.reshape(res, (-1,n_input))

def get_point():
    point = []
    line = get_input()
    point = line.split(' ')
    write_log("Point "+point[0]+" "+point[1]+" recieved")
    if point[1]!="0": set_point(int(point[1]))
    return np.float32(float(point[0]))

def write_log(S):
    Log = open('Log2.txt', 'a')
    Log.write(S+"\n")
    Log.close()

x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

pred = neural_net(x, weights, biases, keep_prob)
result = tf.argmax(pred,1)
cost = tf.Variable(1.0)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-cost)
init = tf.global_variables_initializer()
Log = open('Log2.txt', 'w')
image = np.array([-1, n_input])
training_iters = int(get_input())
with tf.Session() as sess:
    sess.run(init)
    while step < training_iters:
        write_log("Step " + str(step) + " Started")
        if step%10==0:
            p=p*0.99
            write_log("Probability changed to " + str(p))
        image = get_image()
        x = tf.convert_to_tensor(image)
        Log.write("Image recieved")
        r = random.random()
        if r<p:
            action = random.randrange(0,5)
            write_log("Random action")
        else:
            pred = neural_net(x, weights, biases, keep_prob)
            result = tf.argmax(pred, 1)
            action = result.eval()[0]
            write_log("Not random action")
        #print(step)
        write_log("Before send")
        move(action)
        write_log("Move "+str(action)+" sent")
        step = step + 1
        cost = tf.Variable(get_point())
        sess.run(optimizer, feed_dict={x: image, keep_prob: dropout})	
