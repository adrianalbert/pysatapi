

	image_db = lmdb.open(savepath, map_size=map_size)
	move_db(savepath)
	print savepath
	print "Saving %d records (of 100%%):"%len(sources),
	with image_db.begin(write=True) as in_txn:
	    for i,s in enumerate(sources):
	        src,lab = s if len(s)>1 else (s,None)
	        if i % step == 0: print "%d%%"%(i*10/step),
	        # load data for current record
	        img = caffe.io.load_image(src)
	        img = preprocess(img)
	        img_dat = caffe.io.array_to_datum(img, label=lab)
	        write_to_lmdb(image_db, '{:0>10d}'.format(i), img_dat)
	        # in_txn.put('{:0>10d}'.format(i), img_dat.SerializeToString())
	print "done", 
	in_db.close()



def add_data_layer(network_file, source, batch_size=32, phase="TRAIN"):
	# create data layer text
	mean_file = "%s/imagenet_mean.binaryproto"%os.path.dirname(network_file)
	transform_param = dict(mean_file=mean_file) if os.path.exists(mean_file) else None
	args = {"name":"data",
			"top":["data", "label"],
			"source":source, 
			"batch_size":batch_size}

	data_layer = """layer {{
    name: "data"
    type: DATA
    top: "data"
    top: "label"
    data_param {{
        source: "{0}"
        backend: "{1}"
        batch_size: {2}
    }}
    transform_param {{
        scale: {3}
        mean_file: "{4}"
    }}
    include: {{ phase: "{5}" }}
}}""".format(source, )

	# insert data layer text at the top of the network text 
	with open(network_file, "r") as f:
		lines = f.readlines()
	for i,l in enumerate(lines):
		if "layer" in l:
			break
	lines = lines[:i] + str(d.to_proto()).split("\n") + lines[i:]
	return "".join(lines)


class CaffeModel(BaseEstimator):
	''' A lightweight interface to Caffe that has the feel of scikit-learn.
	'''
	def __init__(self, network_file, \
		weights_file=None, initSolver=False, addEval=True):
		self._network_file = network_file
		self._weights_file = weights_file
		self._solver = None  						# for now, just one solver

		if initSolver:
			solver_filename = self.make_solver()
		if weights_file is not None and self._solver is not None:
			self.load_weights(self._weights_file)


	def fit(self, datasource, niter=100, weightSavePath=None, batch_size=32,\
		**kwargs):
		# initialize solver 
		blobs = ('loss', 'acc')
		loss, acc = (np.zeros(niter) for _ in blobs)
		for it in range(niter):
			self._solver.step(1)  # run a single SGD step in Caffe
			loss[it],acc[it] = (self._solver.net.blobs[b].data.copy() \
									for b in blobs)
			if it % disp_interval == 0 or it + 1 == niter:
			    loss_disp='; '.join('%s: loss=%.3f, acc=%2d%%'%\
			    	loss[it], np.round(100*acc[n][it]))
			    print '%3d) %s' % (it, loss_disp)     
		
		# Save the learned weights?
		if weightSavePath is None:
			return loss, acc
		weights_file='%s/weights_train.%s.caffemodel'%\
			(weightSavePath, self._solver.net.name)
		self._solver.net.save(weights_file)
		return loss, acc, weights_file		


	def predict(self, X):
		'''
		Runs forward pass of the network to produce predictions.
		X is a (N,W,H,C) ndarray tensor. 
		'''
		net = self._solver.net


	def make_solver(self, savefile=None, **kwargs):
		solver_filename = define_solver(self._network_file,savefile=savefile, **kwargs)
		self._solver = caffe.get_solver(solver_filename)
		return solver_filename


	def load_weights(self, weights_file):
		self._solver.net.copy_from(weights_file)
