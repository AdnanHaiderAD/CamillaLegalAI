#!/usr/bin/python
#Author:Adnan

import tensorflow as tf
import numpy as np
import os
import argparse
import datetime as dt
import DataPreProcess 
from tensorflow.python.layers.core import Dense

#----------------------------- READ DATA ------------------------------------------------------#
def load_data(filename,n_words=10000):
		#           extract data 
		PreProcessObj = DataPreProcess.DataPreProcess(filename)
		# Create Batches of sentences of equal length for training and target
		trainSet,decodeInputSet,targetSet = PreProcessObj.makeInputAndLabels()
		#       build dictionary
		word_to_id = PreProcessObj.build_vocab(n_words=n_words) 
		reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
		#           Get training, decoder input  and target set
		train_data = PreProcessObj.encodeSentencesToIds(trainSet,word_to_id)
		decoder_inputdata = PreProcessObj.encodeSentencesToIds(decodeInputSet,word_to_id)
		test_data = PreProcessObj.encodeSentencesToIds(targetSet,word_to_id)
		vocabulary = PreProcessObj.VocabularySize 
		num_steps = PreProcessObj.seq_len
		max_len = PreProcessObj.max_len

		#print(train_data)
		print("Number of sentences is ",train_data.shape[0])
		#print("Print decoder targets")
		#print(test_data)
		#print("Print decoder inputs")
		#print(decoder_inputdata)
		print("Print seq lens")
		print(num_steps)
		print(num_steps.shape)
		print("print size of vocabulary")
		print(vocabulary)
		return train_data,decoder_inputdata,test_data, vocabulary, reversed_dictionary, num_steps,max_len

#--------------------------------------  CREATE INPUT PIPELINE------------------------------------#
class Input(object):
		def __init__(self, batch_size,seq_max_len, num_steps, train_data,decoder_inputdata,test_data):
				self.batch_size = batch_size
				self.epoch_size = (train_data.shape[0]//batch_size)
				self.seq_max_len = seq_max_len
				print("epoch size is ")
				print (self.epoch_size)
				self.input_data,self.decoder_input, self.targets,self.num_steps = self.batch_producer(train_data=train_data, decoder_inputdata=decoder_inputdata,test_data=test_data, batch_size=batch_size, num_steps=num_steps)


		def batch_producer(self,train_data,decoder_inputdata,test_data, batch_size, num_steps):
				train_fulldata = tf.convert_to_tensor(train_data, name="training_data",dtype=tf.int32)
				test_fulldata  = tf.convert_to_tensor(test_data, name="test_data",dtype=tf.int32)
				decode_inputfulldata = tf.convert_to_tensor(decoder_inputdata, name="test_data",dtype=tf.int32)
				seq_lenList = tf.convert_to_tensor(num_steps, name="seq_len",dtype=tf.int32)
				
				epoch_size = (train_data.shape[0]//batch_size)
				decoder_inputUsed = decode_inputfulldata[ 0:epoch_size*batch_size, :]
				train_usedData = train_fulldata[0 : epoch_size*batch_size,:]
				test_usedData = test_fulldata[ 0:epoch_size*batch_size, :]
				seq_len = seq_lenList[0:epoch_size*batch_size]

				i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
				x = train_usedData[i * batch_size:(i + 1) * batch_size,:]
				x.set_shape([batch_size, self.seq_max_len])
				decoder_input = decoder_inputUsed[i * batch_size:(i + 1) * batch_size,:]
				decoder_input.set_shape([batch_size, self.seq_max_len])
				y = test_usedData[i * batch_size:(i + 1) * batch_size,:]
				y.set_shape([batch_size, self.seq_max_len])
				num_steps = seq_len [i *batch_size:(i + 1) * batch_size]
				num_steps.set_shape([batch_size])
				return x,decoder_input,y,num_steps


#---------------------------------- MODEL -------------------------------------------------------#
#                       Create the main model
class Model(object):
		def __init__(self, inputObj, is_training, hidden_size, vocab_size, num_layers=1,
								dropout=0.5,useSGD=True):
				self.is_training = is_training
				self.input_obj = inputObj
				self.seq_max_len = inputObj.seq_max_len
				self.batch_size = inputObj.batch_size
				self.num_steps = inputObj.num_steps
				self.hidden_size = hidden_size
				self.vocab_size = vocab_size
				#     Create word2vec embedding layer
				inputs = self.createEmbedding(vocab_size=vocab_size,is_training=is_training,dropout=dropout,input_data= self.input_obj.input_data)
				decoder_inputs = self.createEmbedding(vocab_size=vocab_size,is_training=is_training,dropout=dropout,input_data=self.input_obj.decoder_input)
				#     setup encoder-decoder model
				self.setupEncoder(inputs,num_layers,is_training,dropout)
				output = self.setupDecoder(decoder_inputs,num_layers,is_training,dropout)
				logits = self.setupOutputLayer(output,vocab_size)
				self.setupLossFunction(logits)
				
				# get the prediction accuracy
				#self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
				self.softmax_out = tf.nn.softmax(logits)
				self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
				correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
				self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				if not is_training:
						return
				else:
						self.setupOptimisation(useSGD)  

		#------------------------------------------------------------------------------------    

		###   create the word embeddings for encoder and decoder
		def createEmbedding(self,vocab_size,is_training,dropout,input_data):
				with tf.variable_scope("embeddings",reuse=tf.AUTO_REUSE):
						with tf.device("/cpu:0"):
								embedding = tf.get_variable("word2VecMap",[vocab_size, self.hidden_size])
						inputs = tf.nn.embedding_lookup(embedding,input_data)
						# regularisation    
						if is_training and dropout < 1:
								inputs = tf.nn.dropout(inputs, dropout)
						return inputs 


		#------------------- Setup Encoder ----------------------------#
		def setupEncoder(self,inputs,num_layers,is_training,dropout):
				with tf.variable_scope('encoder') as scope:
						if num_layers > 1:
								self.encoder_init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])
								state_per_layer_list = tf.unstack(self.encoder_init_state, axis=0)
								encoder_initial_state = tuple( [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)])
						else:
								self.encoder_init_state = tf.placeholder(tf.float32, [2, self.batch_size, self.hidden_size])
								state_per_layer = tf.unstack(self.encoder_init_state, axis=0)
								encoder_initial_state = tf.nn.rnn_cell.LSTMStateTuple(state_per_layer[0],state_per_layer[1])

						#  encoder_initial_state = encoder_initial_state[0]
						# create an LSTM encoder cell to be unrolled
						encoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0)
						# add a dropout wrapper if training
						if is_training and dropout < 1:
								encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=dropout)
						if num_layers > 1:
								encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell for _ in range(num_layers)], state_is_tuple=True)
						#   setup encoder
						self.all_encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
							cell=encoder_cell,
							inputs=inputs,
							sequence_length=self.num_steps,
							initial_state=encoder_initial_state,
							dtype=tf.float32)
						# get final state: this is sentence encoding
						encoder_state = tf.unstack(self.encoder_state, axis=0)
						if num_layers > 1:
								self.encoder_final_Layer_state = encoder_state[-1][0]
						else:
								self.encoder_final_Layer_state = self.encoder_state[0]
						
		#------------------------- Setup Decoder ----------------------------------#    
		def setupDecoder(self,inputs,num_layers,is_training,dropout):
				with tf.variable_scope('decoder') as scope:
					# Initial state is last relevant state from encoder
						decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size, forget_bias=1.0)
						if is_training and dropout < 1:
								decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=dropout)

						if num_layers > 1:
								decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell for _ in range(num_layers)], state_is_tuple=True) 
						# setup decoder
						self.all_decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(
											cell=decoder_cell,
											inputs=inputs,
											time_major=False,
											sequence_length=self.num_steps,
											initial_state=self.encoder_state)  
						
						output = tf.reshape(self.all_decoder_outputs, [-1, self.hidden_size])
						return output
			 

		 #---------------------- Setup Output of Encoder-Decoder------------------#   
		def setupOutputLayer(self,output,vocab_size):
				with tf.device("/cpu:0"):
						softmax_w = tf.get_variable("softmax_w",[self.hidden_size, vocab_size])
						softmax_b = tf.get_variable("softmax_b",[vocab_size])
				logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
				#logits = tf.reshape(logits, [self.batch_size, self.seq_max_len, vocab_size])
				return logits

		#------------  Loss function --------------------------------------------#  
		def setupLossFunction(self,logits):
				#loss = tf.contrib.seq2seq.sequence_loss(
				#				logits,
				#				self.input_obj.targets,
				#				tf.ones([self.batch_size, self.seq_max_len], dtype=tf.float32),
				#				average_across_timesteps=False,
				#				average_across_batch=True)

						# Update the cost
				#self.cost = tf.reduce_sum(loss)
				targets_flat = tf.reshape(self.input_obj.targets,[-1])
				losses_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets_flat)
				mask = tf.sign(tf.to_float(targets_flat))
				masked_losses = mask * losses_flat
				masked_losses = tf.reshape(masked_losses,  tf.shape(self.input_obj.targets))
				self.cost = tf.reduce_mean(tf.reduce_sum(masked_losses, reduction_indices=1))

		#------------------- Optimisation ------------------------------#
		def setupOptimisation(self,useSGD):
				self.learning_rate = tf.Variable(0.0, trainable=False)
				tvars = tf.trainable_variables()
				#clip gradients 
				grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
				if (useSGD):
					optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
				else:
					optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.train_op = optimizer.apply_gradients(
						zip(grads, tvars),
						global_step=tf.contrib.framework.get_or_create_global_step())
				# self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

				self.new_lr = tf.placeholder(tf.float32, shape=[])
				self.lr_update = tf.assign(self.learning_rate, self.new_lr)

		def assign_lr(self, session, lr_value):
				session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------TRAIN MODEL -----------------------------------------------------------------------------
def train(train_data,
	decoder_inputdata,
	test_data,
	vocabulary, 
	num_layers,
	hidden_size, 
	num_epochs, 
	batch_size,
	seq_max_len,
	num_steps,
	model_dir,
	useSGD = True,
	dropout = 0.5,
	learning_rate=1.0, 
	max_lr_epoch=10, 
	lr_decay=0.93, 
	print_iter=20,
	intermediate_model = None
	):
		# setup data and models
		training_input = Input(batch_size=batch_size, num_steps=num_steps,seq_max_len = max_len,train_data=train_data,decoder_inputdata=decoder_inputdata,test_data=test_data)
		m = Model(training_input, is_training=True, hidden_size=hidden_size, vocab_size=vocabulary,num_layers=num_layers,dropout =dropout)
		init_op = tf.global_variables_initializer()
		orig_decay = lr_decay
		with tf.Session() as sess:
				# start threads
				sess.run([init_op])
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord)
				saver = tf.train.Saver()
				#restore interemediate model
				if intermediate_model is not None:
					saver.restore(sess, intermediate_model)
					print("intermediate_model loaded")
				for epoch in range(num_epochs):
						#assign new learning rate using exponential decay
						new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
						m.assign_lr(sess, learning_rate * new_lr_decay)
						print("Current learning rate ",learning_rate * new_lr_decay)
						if num_layers > 1:
							current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
						else:
							current_state = np.zeros((2, batch_size, m.hidden_size))
						curr_time = dt.datetime.now()
						for step in range(training_input.epoch_size):
								# cost, _ = sess.run([m.cost, m.optimizer])
								if step % print_iter != 0:
										cost, _, current_state = sess.run([m.cost, m.train_op, m.encoder_state],
																		feed_dict={m.encoder_init_state: current_state})
										#print(final_state)
								else:
										seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
										curr_time = dt.datetime.now()
										cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.encoder_state, m.accuracy],
																				feed_dict={m.encoder_init_state: current_state})
										print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
														step, cost, acc, seconds))

						# save a model checkpoint
						print("Saving checkpoint");
						saver.save(sess,os.path.join(model_dir,str(epoch) +'model_intermediate.ckpt'))
				# do a final save
				print("Saving final model ")
				print(os.path.join(model_dir,'trained_model.ckpt'))
				saver.save(sess,os.path.join(model_dir,'trained_model.ckpt'))
				# close threads
				coord.request_stop()
				coord.join(threads)

#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------ENCODE SENTENCES WITH TRAINED MODEL -----------------------------------------------------------------------------


def test(train_data,
	decoder_inputdata,
	test_data,
	vocabulary, 
	num_layers,
	hidden_size, 
	batch_size,
	seq_max_len,
	num_steps,
	model_name):
	
		test_input = Input(batch_size=batch_size, num_steps=num_steps,seq_max_len = max_len,train_data=train_data,decoder_inputdata=decoder_inputdata,test_data=test_data)
		m = Model(test_input, is_training=False, hidden_size=hidden_size, vocab_size=vocabulary,num_layers=num_layers)
		with tf.Session() as sess:
				# start threads
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord)
				saver = tf.train.Saver()
				# Model assumes LSTM structure
				current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
				#Sentence_encode_output
				encoded_output =[]
				if num_layers > 1:
						current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
				else:
						current_state = np.zeros((2, batch_size, m.hidden_size))
				#restore saved model
				saver.restore(sess, model_name)
				curr_time = dt.datetime.now()
				print("Number of batches")
				print(test_input.epoch_size)
				for step in range(test_input.epoch_size):
						current_state, final_state = sess.run([m.encoder_state,m.encoder_final_Layer_state],
																											feed_dict={m.encoder_init_state: current_state})
						print
						print(step)
						print(final_state)
						encoded_output.extend(final_state)
			 	print("Output encoded sentences")
				print(encoded_output)		
				# close threads
				coord.request_stop()
				coord.join(threads)

			
#------------------------------------------------------------------------------------------------------------#



#--------- Default Data path ------------
data_path = "/Users/mah90/TensorflowCode/Code/data/sentences.txt"
model_dir = "/Users/mah90/TensorflowCode/Encoder_DecoderCode/Camilla_codeBase/models"
model_name = "/Users/mah90/TensorflowCode/Encoder_DecoderCode/Camilla_codeBase/models/trained_model.ckpt"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
parser.add_argument('--model_dir', type=str, default=model_dir, help='Dir to save  models from')
parser.add_argument('--model_name', type=str, default=model_name, help='Encoder model to load')
args = parser.parse_args()


# ---------------    
if args.data_path:
		data_path = args.data_path

if args.model_dir:    
		model_dir = args.model_dir

if args.model_name:    
		model_name = args.model_name

train_data,decoder_inputdata,test_data, vocabulary, reversed_dictionary, num_steps,max_len = load_data(data_path)

if args.run_opt == 1:
		train(train_data,
		decoder_inputdata,
		test_data, 
		vocabulary, 
		num_layers = 1, 
		hidden_size = 128,
		num_epochs = 20, 
		batch_size = 10,
		seq_max_len = max_len,
		num_steps= num_steps,
		model_dir = model_dir,
		useSGD = False,
		learning_rate = 0.5, 
		dropout = 1.0,
		max_lr_epoch = 10, 
		lr_decay = 0.93,
		print_iter = 100,
		intermediate_model = model_name
		)
else:
		test(train_data,
		decoder_inputdata,
		test_data, 
		vocabulary, 
		num_layers=1, 
		hidden_size= 128,
		batch_size=10,
		seq_max_len = max_len,
		num_steps= num_steps,
		model_name = model_name,
		)

		
# Example run: python Encoder.py 1 --data_path  /Users/mah90/TensorflowCode/Encoder_DecoderCode/Camilla_codeBase/data/sentences.txt  --model_dir /Users/mah90/TensorflowCode/Encoder_DecoderCode/Camilla_codeBase/models


