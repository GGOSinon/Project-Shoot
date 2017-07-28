import tensorflow as tf
import subprocess
learning_rate = 0.01
training_iters = 200000
mem_size = 128

n_input = 128*128
n_classes = 5
dropout = 0.75

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
    conv1 = norm(conv1)
    #print(conv1.shape) 
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    conv2 = norm(conv2)
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    conv3 = norm(conv3)

    fc1 = fcl(conv3, weights['wd1'], biases['bd1'])
    
    out = fcl(fc1, weights['out'], biases['out'])
    return out

def move(x):
    if x==1 : S="forward"
    elif x==2 : S="backward"
    elif x==3 : S="left"
    elif x==4 : S="right"
    else: return
    print(S+"\r\n")

def get_image():
    line = input()
    return line

def get_point():
    line = input()
    return float(line)

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

pred = neural_net(x, weights, biases, keep_prob)
cost = tf.Variable(1.0)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < training_iters:
        x = get_image()
        pred = neural_net(x, weights, biases, keep_prob)
        action = tf.argmax(pred, 1)
        move(action)
        sess.run(optimizer, feed_dict={x: x, y: y, keep_prob: dropout})
