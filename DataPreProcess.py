#!/usr/bin/python
#Author:Adnan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import re
import codecs
from collections import Counter
import numpy as np


class DataPreProcess:

	#----------------------- Read Data and extract senstences and all words -----------------------#
	def extractData(self,filename):
		readData = []
		#--------- Create Sentence list ----------------#
		with codecs.open(filename, "r+",encoding='UTF-8') as f:
			readData = f.readlines()
		if len(readData) == 0:
			print(" Error with reading")
			exit(0)
		sentences =[]
		data = []
		for line in readData:
			line = re.sub(r"([.,!?\"':;)(])", "", line.strip('\n'))
			if len(line) == 0:
				continue  # handle empty lines
			line = re.sub(r"'s", "", line) # this line doesnt seem to work properly
			words = line.split()
			sentences.append(words)
			#----------- Creat word list ----------------------#
			data.extend(words)
		return sentences,data	

	#-------------------- Create Batches of sentences of equal length for training and target -----------#
	def makeInputAndLabels(self,sentences=None):
		if sentences is None:
			sentences = self.sentences
		max_len = max(len(sentence) for sentence in sentences)
		self.max_len = max_len
		train_set =[]
		decoder_inputs =[]
		target_set = []
		seq_len =[]
		for sentence in sentences:
			target = sentence[:]
			target.reverse()
			print (sentence)
			print (target)
			sen_len = len(sentence)
			seq_len.append(sen_len)
			if sen_len == max_len :
				train_set.append(sentence)
				target_set.append(target)
				decoder_input = ['_GO']+ target[:-1]
				decoder_inputs.append(decoder_input)
			else :
				diff_len = max_len - len(sentence)
				print(max_len)
				print(len(sentence))
				padding = ['<\s>']*diff_len
				sentence_extended =  sentence+padding
				train_set.append(sentence_extended)
				target_extended = target+padding
				decoder_input = ['_GO']+ target[:-1]+ padding
				target_sentence = target_extended
				decoder_inputs.append(decoder_input)
				target_set.append(target_sentence)
		self.seq_len = np.array(seq_len, np.int32) # collect seq lengths	
		return train_set,decoder_inputs,target_set		

	#-------------- Encode sentences in B X T matrix---------------#
	def encodeSentencesToIds(self,sentence_dataset,word_to_id):
		output =[[word_to_id[word] for word in sentence] for sentence in sentence_dataset]
		output = np.asarray(output)
		return np.asmatrix(output)

	#--------------------------- BUILD Dictionary ---------------------------------------#
	def build_vocab(self,data=None,n_words=10000):
		if data is None:
			data = self.data
		counter = Counter(data)
		counter = counter.most_common(n_words)
		count_pairs = sorted(counter, key=lambda x: (-x[1], x[0]))
		words, _ = list(zip(*count_pairs))
		print(words)
		#--Add special symbols to word list		
		_PAD = '<\s>'
		_GO = '_GO'
		_START_VOCAB = [_PAD, _GO]
		vocab_list = _START_VOCAB + list(words)
		word_to_id = dict([(x,y) for (y,x) in enumerate(vocab_list)])
		self.VocabularySize = len(vocab_list)
		return word_to_id

	def __init__(self,filename, n_words=10000):
		self.sentences,self.data = self.extractData(filename)
	


	



