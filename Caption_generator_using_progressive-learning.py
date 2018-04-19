from os import listdir
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
# keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add

from pickle import load
#import pydot

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# store
		descriptions[image_id] = ' '.join(image_desc)
	return descriptions

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)[0]
	# get image id
	image_id = filename.split('/')[-1].split('.')[0]
	return image, image_id

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc, image):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	# integer encode the description
	seq = tokenizer.texts_to_sequences([desc])[0]
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# select
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	Ximages, XSeq, y = np.array(Ximages), np.array(XSeq), np.array(y)
	return [Ximages, XSeq, y]

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, tokenizer, max_length):
	# loop for ever over images
	directory = 'Flicker8k_Dataset'
	while 1:
		for name in listdir(directory):
			# load an image from file
			filename = directory + '/' + name
			image, image_id = load_photo(filename)
			# create word sequences
			desc = descriptions[image_id]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
			yield [[in_img, in_seq], out_word]
            
def build_model(vocab_size,max_length):
    
    '''Feature Extraction Model'''
    input_image = Input(shape=(4096,))
    dropout_layer = Dropout(0.5)(input_image)
    dnn_layer = Dense(256,activation = 'relu')(dropout_layer)
    
    '''Sequence Model'''
    input_text = Input(shape = (max_length,))
    embedding_layer = Embedding(vocab_size,256,mask_zero = True)(input_text)
    dropout_layer2 = Dropout(0.5)(embedding_layer)
    lstm_layer = LSTM(256)(dropout_layer2)
    
    '''Decoder Model'''
    decoder1 = add([dnn_layer,lstm_layer])
    decoder2 = Dense(256,activation = 'relu')(decoder1)
    outputs = Dense(vocab_size,activation = 'softmax')(decoder2)
    
    '''Model'''
    model = Model(inputs = [input_image,input_text],output = outputs)
    
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam')
    
    print (model.summary())
    return model
    #plot_model(model,to_file = 'model.png',show_shapes = True)
    

# load mapping of ids to descriptions
descriptions = load_clean_descriptions('image_description.txt')
# integer encode sequences of words
tokenizer = create_tokenizer(descriptions)
# pad to fixed length
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)

# test the data generator
# define the model
vocab_size = len(tokenizer.word_index) + 1
model = build_model(vocab_size, max_length)

print ("Model Built")
# define checkpoint callback

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

print ("Check point defined")

print ("Data Generator started")
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

# define model
# ...
# fit model
print ("Fitting the model")
model.fit_generator(data_generator(descriptions, tokenizer, max_length),epochs = 10, steps_per_epoch=70000,epochs = 10,verbose = 2 ,callback = [checkpoint])



