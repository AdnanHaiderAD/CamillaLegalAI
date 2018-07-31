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
	def extractData(self,filename,batchSize):
			readData = []
			#--------- Create Sentence list ----------------#
			with codecs.open(filename, "r+",encoding='UTF-8') as f:
					readData = f.readlines()
			if len(readData) == 0:
					print(" Error with reading")
					exit(0)
			
			data = []
			numOfSent = 0
			sentenceSet =[]
			numberofChunks = 0
			maxChunkNum = 0
			for line in readData:
					line = re.sub(r"([.,!?\"':;)(])", "", line.strip('\n'))
					if len(line) == 0:
							continue  # handle empty lines
					line = re.sub(r"'s", "", line) # this line doesnt seem to work properly
					line = line.lower()
					words = line.split()
					data.extend(words) 
					numOfSent+=1
					# Restricting the length of sentences to a max ceiling length. This helps alleviate vanishing gradients
					sentence =[ ]
					chunkNum  = 0
					if len(words) > 2:
							chunks = len(words) //2
							remainder = len(words) % 2
							words_batch = words[0:chunks*2]
							if remainder > 0:
								words_remain = words[chunks*2:]
								chunkNum +=1
							chunkNum +=chunks
							numberofChunks +=chunkNum
							if chunkNum > maxChunkNum:
								maxChunkNum = chunkNum
							if chunks >1:
									for i in range(chunks):
										segment = words_batch[i*2:(i+1)*2]
										sentence.append(segment)
							else:
									sentence.append(words_batch)
							if remainder >0:    	
								sentence.append(words_remain)    	
							sentenceSet.append(sentence)  
							continue
					numberofChunks+=1        
					sentence.append(words)        
					sentenceSet.append(sentence)        
			# Pack sentences such that chunks belonging to a sentence occupy the same position in batches		
			sentence_id = [None]*len(sentenceSet)
			sentences =[None]*numberofChunks
			padding = [None]*(batchSize*maxChunkNum)
			sentences = sentences + padding
			count = 0
			currentSentence = sentenceSet[count]
			j = 0
			for sentence in sentenceSet:
				while sentences[j] is not None:
					j+=1
				for chunk_index, chunk in enumerate(sentence):
					sentences[j + chunk_index*batchSize] = chunk
				sentence_id[count] = j + (len(sentence)-1)*batchSize
				count+=1	 	

			for i in range(len(sentences)):
				if sentences[i] is None:
					sentences[i] = ['<\s>']	
			return sentences,data,sentence_id


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
			sen_len = len(sentence)
			seq_len.append(sen_len)
			if sen_len == max_len :
				train_set.append(sentence)
				target_set.append(target)
				decoder_input = ['_GO']+ target[:-1]
				decoder_inputs.append(decoder_input)
			else :
				diff_len = max_len - len(sentence)
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
		#--Add special symbols to word list		
		_PAD = '<\s>'
		_GO = '_GO'
		_START_VOCAB = [_PAD, _GO]
		vocab_list = _START_VOCAB + list(words)
		word_to_id = dict([(x,y) for (y,x) in enumerate(vocab_list)])
		self.VocabularySize = len(vocab_list)
		return word_to_id

	def __init__(self,filename,batchSize, n_words=10000):
		self.sentences,self.data,self.sentence_id = self.extractData(filename,batchSize)
	


	



