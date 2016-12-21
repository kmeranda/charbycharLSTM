import sys, getopt
from __future__ import print_function, division
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq
import numpy as np
#import keras

class LSTMModel():
	def __init__(self, vocab, batch_size=64, seq_length=50):
	# init net variables
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.rnn_size = 512
		self.vocab_size = len(vocab) 	# calculated later
		self.vocab = vocab
		self.num_layers = 2
		self.epochs = 20
		self.grad_clip = 5.0
		self.training_iters = 100000
		self.display_step = 10
		self.size = self.batch_size*self.seq_length	# total size of input data for graph
		self.build_graph()
	
	def build_graph(self):
		# graph input/output
		print('Build graph')
		self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='input_placeholder')	# inputs
		self.targets = tf.placeholder(tf.int64, [self.batch_size, self.seq_length])	# outputs
		# pick rnn function (LSTM)
		cell_fn = rnn_cell.BasicLSTMCell
		cell = cell_fn(self.rnn_size)
		self.cell = rnn_cell.MultiRNNCell([cell] * self.num_layers)	# make multilayer
		with tf.variable_scope('lstm'):
			# character embedding of input
			embedding = tf.get_variable('embedding', [self.vocab_size, self.rnn_size])
			input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)	# dims now [batch_size, seq_length, rnn_size]
			# weights and bias
			self.weights = { 'out': tf.get_variable('weights', [ self.rnn_size, self.vocab_size ]) }
			self.biases = { 'out': tf.get_variable('biases', [ self.vocab_size ]) }
			# unfold rnn
			inputs = tf.split(1, self.seq_length, input_embeddings)
			outputs = []
			self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
			state = self.initial_state
			for i, inp in enumerate(inputs):
				# reusing variables from called LSTM variables from first time-step
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				inp = tf.squeeze(inp, [1])
				output, state = self.cell(inp, state)
				outputs.append(output)
		output = tf.reshape(tf.concat(1, outputs), [-1, self.rnn_size])
		# calculate softmax logits and loss
		self.logits = tf.add(tf.matmul(output, self.weights['out']), self.biases['out'])
		self.probs = tf.nn.softmax(self.logits)
		loss_c = seq2seq.sequence_loss_by_example([self.logits],
			[tf.reshape(self.targets, [-1])],
			[tf.ones([self.batch_size * self.seq_length])],
			self.vocab_size)
		self.cost = tf.reduce_sum(loss_c) / self.batch_size / self.seq_length
		self.final_state = state
		# gradient clipping
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
		# define training optimizer
		optimizer = tf.train.AdamOptimizer()
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
	
	def train_model(self, X_train, Y_train):
		# to evaluate the model
		correct_pred = tf.equal(tf.argmax(self.logits,1), tf.reshape(self.targets, [-1]))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		# initialize all variables
		init = tf.initialize_all_variables()
		indices = np.arange(len(X_train))
		# initialize saver
		saver = tf.train.Saver()
		num_batches_per_epoch = len(X_train) // self.batch_size
		print('Training')
		# launch the graph
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			# try to load saved model
			ckpt = tf.train.get_checkpoint_state('model/')
			if not ckpt or not ckpt.model_checkpoint_path:
				# train for 20 epochs
				for i in range(self.epochs):
					np.random.shuffle(indices)
					X_train = X_train[indices]
					Y_train = Y_train[indices]
					
					# go through all batches in train set
					for j in xrange(num_batches_per_epoch):
						# define current batch
						batch_x = X_train[j * self.batch_size: (j + 1) * self.batch_size]
						batch_y = Y_train[j * self.batch_size: (j + 1) * self.batch_size]
		
						# define feed with current batch
						feed = {
							self.input_data: batch_x,
							self.targets: batch_y,
						}
						loss, acc = sess.run([self.cost, accuracy], feed)
						if j % self.display_step == 0: # only print select batch accuracies
							s = 'Epoch ' + str(i) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc)
							print(s)
					saver.save(sess, 'model/my-model', global_step=i)
			else:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Successfully loaded model')
			if self.seq_length == 1 and self.batch_size == 1:
				s = self.generate(sess)
				print(s)
		
	def generate(self, sess):
		length=5000
		inp='X: '
		chars = { v: k for k, v in self.vocab.iteritems() }
		ids = self.vocab
		state = sess.run(self.cell.zero_state(1, tf.float32))
		# set up state of model to gen after input
		for char in inp[:-1]:
			x = np.zeros((1,1))
			x[0,0] = ids[char]
			feed = { self.input_data: x, self.initial_state: state }
			[state] = sess.run([self.final_state], feed)
		
		s = inp
		curr = inp[-1]
		# generate sample
		for n in range(length):
			x = np.zeros((1,1))
			x[0,0] = ids[curr]
			feed = {self.input_data: x, self.initial_state: state }
			[samp_probs, state] = sess.run([self.probs, self.final_state], feed)
			p = samp_probs[0]
			sample = self.weighted_pick(p, curr)	# pick next character index
			nxt = chars[sample]	# get next character from id
			s += nxt	# add to output
			curr = nxt	# prep for next iteration
		return s	# return generated sample

	def weighted_pick(self, weights, char):
		t = np.cumsum(weights)
		s = np.sum(weights)
		if char == ' ':
			return int(np.searchsorted(t, np.random.rand(1)*s))
		else:
			return np.argmax(weights)

def main(argv):
	inputfile = 'train.txt'
	outputfile = 'generated_songs.txt'
	# take filenames from arguments
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	########
	# Data #
	########
	seq_length = 50
	# create character-id mapping
	data = open(inputfile).read()
	outfile = open(outputfile, 'w+')
	alphabets = set(data)
	vocab = {}
	i = 0
	for ch in alphabets:
		vocab[ch] = i
		i += 1
	vocab_size = len(alphabets)
	chars = { v: k for k, v in vocab.iteritems() }
	# create train data from file
	print('Process data')
	X_train = []
	Y_train = []
	for i in xrange(len(data) - seq_length - 1):
		x = np.array(map(vocab.get, data[i: i + seq_length]))
		y = np.array(map(vocab.get, data[i + 1: i + 1 + seq_length]))
		X_train.append(x)
		Y_train.append(y)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	#########
	# Model #
	#########
	with tf.variable_scope('my_model', reuse=False):
		model = LSTMModel(vocab)
		#model.train_model(X_train, Y_train)
	with tf.variable_scope('my_model', reuse=True):
		sample_model = LSTMModel(vocab, seq_length=1, batch_size=1)
		#sample_model.train_model(X_train, Y_train)
	# to evaluate the model
	correct_pred = tf.equal(tf.argmax(model.logits,1), tf.reshape(model.targets, [-1]))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	# initialize all variables
	init = tf.initialize_all_variables()
	indices = np.arange(len(X_train))
	# initialize saver
	saver = tf.train.Saver()
	num_batches_per_epoch = len(X_train) // model.batch_size
	#########
	# Train #
	#########
	print('Training')
	# launch the graph
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		# try to load saved model
		ckpt = tf.train.get_checkpoint_state('model/')
		if not ckpt or not ckpt.model_checkpoint_path:
			# train for 20 epochs
			for i in range(model.epochs):
				np.random.shuffle(indices)
				X_train = X_train[indices]
				Y_train = Y_train[indices]
				
				# go through all batches in train set
				for j in xrange(num_batches_per_epoch):
					# define current batch
					batch_x = X_train[j * model.batch_size: (j + 1) * model.batch_size]
					batch_y = Y_train[j * model.batch_size: (j + 1) * model.batch_size]
	
					# define feed with current batch
					feed = {
						model.input_data: batch_x,
						model.targets: batch_y,
					}
					loss, acc, _ = sess.run([model.cost, accuracy, model.train_op], feed)
					if j % model.display_step == 0: # only print select batch accuracies
						s = 'Epoch ' + str(i) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc)
						print(s)
				saver.save(sess, 'model/my-model', global_step=i)
				s = sample_model.generate(sess)
				print(s)
				outfile.write(s)
		else:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Successfully loaded model')
		s = sample_model.generate(sess)
	print(s)
	outfile.write(s)

def get_seq(line, vocab):
	seq = []
	for char in line:
		seq.append(vocab[char])
	return seq 

if __name__ == '__main__':
	main(sys.argv[1:])
