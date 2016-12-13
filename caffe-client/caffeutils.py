# numeric packages
import numpy as np

# filesystem and OS
import sys, os, time
import glob
import tempfile

# computer vision
import skimage

# ML
import sklearn
from sklearn.base import BaseEstimator

# Caffe-related
import caffe
import lmdb 
from caffe.proto import caffe_pb2


class CaffeModel(BaseEstimator):
	''' A lightweight interface to Caffe that has the feel of scikit-learn.
	'''

	def __init__(self, train_net_file, test_net_file=None, \
		weights_file=None, initSolver=False):
		self._train_net_file = train_net_file
		self._test_net_file = test_net_file
		self._weights_file = weights_file

		self.make_solver()
		if weights_file is not None:
			self._solver.net.copy_from(weights)



	def fit(self, dataSouce, nIter=100, **kwargs):
		solver = self.make_solver(self._test_net_file)
		pass


	def predict(self, dataSource):
		pass


	def make_solver(self, savefile=None, **kwargs):
		solver_filename = define_solver(self._train_net_file, test_net=self._test_net_file, savefile=savefile)
		self._solver = caffe.get_solver(solver_filename)
		return solver_filename


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


def save_images_to_lmdb(sources, savepath="./", imgSize=(3,500,500)):
	''' 
	Create LMDB database from images taken from a list of sources.
	'''
	# estimate size of database
	X = np.zeros((1,)+imgSize, dtype=np.uint8)
	map_size = int(X.nbytes * len(sources) * 5)
	step = len(sources) / 10

	if not os.path.exists(savepath):
		os.makedirs(savepath)

	in_db = lmdb.open(savepath, map_size=map_size)
	print savepath
	print "Saving %d records (of 100%%):"%len(sources),
	with in_db.begin(write=True) as in_txn:
	    for i,s in enumerate(sources):
	        src,lab = s if len(s)>1 else (s,None)
	        if i % step == 0: print "%d%%"%(i*10/step),
	        # load data for current record
	        img = skimage.io.imread(src)
	        img = img[:,:,::-1]
	        img = img.transpose((2,0,1))
	        img_dat = caffe.io.array_to_datum(img)
	        if lab is not None: img_dat.label = lab
	        in_txn.put('{:0>10d}'.format(i), img_dat.SerializeToString())
	in_db.close()


def define_solver(train_net, test_net=None, savefile=None, **kwargs):
	'''
	Define solver parameters and save to file.
	'''
	s = standard_solver_parameters(train_net)
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
	s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'

	# Train on the GPU if availble! Using the CPU to train large networks is very slow.
	s.solver_mode = caffe_pb2.SolverParameter.CPU

	return s


