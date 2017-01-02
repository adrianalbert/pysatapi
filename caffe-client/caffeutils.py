# numeric packages
import numpy as np
import math
import random

# filesystem and OS
import sys, os, time
import glob
import tempfile
import string
import shutil

# computer vision
import skimage

# ML
import sklearn
from sklearn.base import BaseEstimator

# Caffe-related
import caffe
import lmdb 
from caffe.proto import caffe_pb2

def make_imagenet_data_transformer(data_shape):
	# load the mean ImageNet image (as distributed with Caffe) for subtraction
	mean_rgb = np.array([107.4072, 107.8797, 103.334])
	mean_bgr = np.array([103.334, 107.8797, 107.4072])

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': (1,)+data_shape})
	transformer.set_transpose('data', (2,0,1))        # move image channels to outermost dimension
	transformer.set_raw_scale('data', 255)            # rescale from [0, 1] to [0, 255]
	transformer.set_mean('data', mean_rgb)            # subtract the dataset-mean value in each channel
	transformer.set_channel_swap('data', (2,1,0))     # swap channels from RGB to BGR
	return transformer


def eval_net(network_file, weights_file, test_iters=10):
    test_net = caffe.Net(network_file, weights_file, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy


def extract_features(net, imgPaths, \
	batchSize=128, imgSize=(400,400,3), \
	feature_layer="conv7", prob_layer="probs"):
    '''
    Runs the forward pass of the neural net on every image in the batch. 
    '''
    # note that this specific architecture expects batches of 32 images of size 400x400x3
    numImages = len(imgPaths)
    batchSize = np.min([batchSize, numImages])
    numBatches = math.ceil(numImages * 1.0 / batchSize)
    net.blobs['data'].reshape(batchSize, imgSize[2], imgSize[0], imgSize[1])
    batches = np.array_split(imgPaths, numBatches)
    transformer = make_imagenet_data_transformer(net.blobs['data'].data.shape)

    raw_features = []
    probs = []
    print "Batch (of %d):"%len(batches),
    for iBatch, curPaths in enumerate(batches):
        print "%d,"%iBatch,
        batch = np.vstack([np.expand_dims(transformer.preprocess("data", caffe.io.load_image(p)),0) \
                           for p in curPaths])
        if len(batch) != batchSize:
            net.blobs['data'].reshape(len(batch), imgSize[2], imgSize[0], imgSize[1])
        net.blobs['data'].data[...] = batch
        output = net.forward()
        raw_features.append(net.blobs[feature_layer].data)
        probs.append(output[prob_layer])
    print "done"
    # get final features
    raw_features = np.vstack(raw_features)
    n, f, h, w = raw_features.shape
    features = raw_features.reshape(n, f, h*w)
    features = np.mean(features, axis=2)
    # get predictions
    probs = np.squeeze(np.vstack(probs))
    return features, probs


def define_solver(train_net, savefile=None, **kwargs):
	'''
	Define solver parameters and save to file.
	'''
	s = standard_solver_parameters(train_net)

	# make directory for snapshots 
	if hasattr(s, 'snapshot_prefix') and not os.path.exists(s.snapshot_prefix):
		os.makedirs(s.snapshot_prefix)

	for k,v in kwargs:
		s.__setattr__(k, v)
	# Write the solver to a temporary file and return its filename.
	f = open(savefile, "w") if savefile is not None \
		else tempfile.NamedTemporaryFile(delete=False)
	f.write(str(s))
	fname = f.name 
	f.close()
	return fname


def standard_solver_parameters(train_net_path, test_net_path=None):
	'''
	Define standard parameters for solver used in Caffe.
	'''
	s = caffe_pb2.SolverParameter()

	# Specify locations of the train and (maybe) test networks.
	s.train_net = train_net_path
	if test_net_path is not None:
	    s.test_net.append(test_net_path)
	    s.test_interval = 1000  # Test after every 1000 training iterations.
	    s.test_iter.append(100) # Test on 100 batches each time we test.

	# The number of iterations over which to average the gradient.
	# Effectively boosts the training batch size by the given factor, without
	# affecting memory utilization.
	s.iter_size = 1

	s.max_iter = 100000     # # of times to update the net (training iterations)

	# Solve using the stochastic gradient descent (SGD) algorithm.
	# Other choices include 'Adam' and 'RMSProp'.
	s.type = 'SGD'

	# Set the initial learning rate for SGD.
	s.base_lr = 0.001

	# Set `lr_policy` to define how the learning rate changes during training.
	# Here, we 'step' the learning rate by multiplying it by a factor `gamma`
	# every `stepsize` iterations.
	s.lr_policy = 'step'
	s.gamma = 0.1
	s.stepsize = 20000

	# Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
	# weighted average of the current gradient and previous gradients to make
	# learning more stable. L2 weight decay regularizes learning, to help prevent
	# the model from overfitting.
	s.momentum = 0.9
	s.weight_decay = 5e-4

	# Display the current training loss and accuracy every 1000 iterations.
	s.display = 1000

	# Snapshots are files used to store networks we've trained.  Here, we'll
	# snapshot every 10K iterations -- ten times during training.
	s.snapshot = 10000
	s.snapshot_prefix = './snapshots/'

	# Train on the GPU if availble! Using the CPU to train large networks is very slow.
	s.solver_mode = caffe_pb2.SolverParameter.CPU

	return s


def preprocess(img):
    mean_rgb = np.array([107.4072, 107.8797, 103.334])
    out = img * 255
    out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
    for i in range(2):
    	out[:,:,i] -= mean_rgb[i]
    return out.astype(np.uint8)


def save_images_to_lmdb(sources, savepath="./", imgSize=(3,500,500)):
	''' 
	Create LMDB database from images taken from a list of sources.
		(source_path, <label>)
	'''
	# estimate size of database
	X = np.zeros((1,)+imgSize, dtype=np.uint8)
	map_size = int(X.nbytes * len(sources) * 5)
	step = len(sources) / 10

	transformer = make_imagenet_data_transformer(imgSize)

	def write_to_lmdb(db, key, value):
		"""
		Write (key,value) to db
		"""
		success = False
		while not success:
			txn = db.begin(write=True)
			try:
			    txn.put(key, value)
			    txn.commit()
			    success = True
			except lmdb.MapFullError:
			    txn.abort()
			    # double the map_size
			    curr_limit = db.info()['map_size']
			    new_limit = curr_limit*2
			    print '>>> Doubling LMDB map size to %sMB ...'%(new_limit>>20,)
			    db.set_mapsize(new_limit) # double it

	if not os.path.exists(savepath):
		os.makedirs(savepath)

	image_db = lmdb.open(savepath, map_size=map_size) 
	print savepath
	print "Saving %d records (of 100%%):"%len(sources),
	for i,s in enumerate(sources):
	    src,lab = s if len(s)>1 else (s,None)
	    if i % step == 0: print "%d%%"%(i*10/step),
	    # load data for current record
	    img = caffe.io.load_image(src)
	    img = transformer.preprocess("data", img)
	    img_dat = caffe.io.array_to_datum(img.astype(int).astype(np.uint8), label=lab)
	    write_to_lmdb(image_db, '{:0>10d}'.format(i), img_dat.SerializeToString())
	    # in_txn.put('{:0>10d}'.format(i), img_dat.SerializeToString())
	print "done"
	image_db.close()


# image_datum = caffe.io.array_to_datum( transformed_image, label )
# write_to_lmdb(image_db, str(itr), image_datum.SerializeToString())


def read_binaryproto_file(filename):
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(filename, 'rb').read()
	blob.ParseFromString(data)
	arr = np.array(caffe.io.blobproto_to_array(blob))
	out = arr[0]
	return arr


def array_to_binaryproto_file(arr, filename):
	blob = caffe.io.array_to_blobproto(arr)
	with open(filename,'wb') as f:
		f.write(blob.SerializeToString())


def parse_prototxt_file(filename):
	_net = caffe.proto.caffe_pb2.NetParameter()
	with open(filename) as f:
	    google.protobuf.text_format.Merge(f.read(), _net)
	return _net


def deprocess_RGB_image(image, imgMean=[123, 117, 104]):
	''' 
	Caffe performs some transformations to images for training.
	This restores the original RGB image (e.g., for plotting).
	'''
	image = image.copy()              # don't modify destructively
	image = image[::-1]               # BGR -> RGB
	image = image.transpose(1, 2, 0)  # CHW -> HWC
	image += imgMean		          # (approximately) undo mean subtraction
	# clamp values in [0, 255]
	image[image < 0], image[image > 255] = 0, 255
	# round and cast from float32 to uint8
	image = np.round(image)
	image = np.require(image, dtype=np.uint8)
	return image

