# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 06:32:56 2018

@author: Saurabh
"""

import string
from pickle import load
import pickle
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint





def load_doc(filename):
    file = open(filename,'r')
    image_map = dict()
    text = file.read()
    file.close()
    desc_list = text.split('\n')
    for desc in desc_list:
        token = desc.split()
        image_Description = ' '.join(token[1:])
        if len(token) < 2:
            continue
        image_id = str(token[0].split('.')[0])
        if image_id in image_map.keys():
            image_map[image_id].append(image_Description)
        else:
            image_map[image_id] = list()
            image_map[image_id].append(image_Description)
    return image_map
            
def clean_description(image_map):
    table = str.maketrans('','',string.punctuation)
    for key,desc_list in image_map.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            
            word_list = desc.split()
            #lowercase
            word_list = [w.lower() for w in word_list]
            #removing punctuation
            word_list = [w.translate(table) for w in word_list]
            #removing word like s and a
            word_list = [w for w in word_list if len(w)>1]
            #removing numbers
            word_list = [w for w in word_list if w.isalpha()]
            
            desc_list[i] = ' '.join(word_list)
            
def make_vocabulary(image_map):
    vocabulary = set()
    for key,desc_list in image_map.items():
        for i in desc_list:
            vocabulary.update(i.split())
    return vocabulary

def save_description(image_map,filename):
    lines = list()
    for key,desc_list in image_map.items():
        for desc in desc_list:
            lines.append(key+' '+desc)
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()

def identifier_set(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    dataset = set()
    for line in text.split('\n'):
        image_id = line.split('.')[0]
        if(len(line))<1:
            continue
        dataset.add(image_id)
    return dataset

def sequence_dict_description(filename,identifiers):
    file = open(filename,'r')
    text = file.read()
    file.close()
    descriptions = dict()
    for line in text.split('\n'):
        tokens = line.split()
        image_id = tokens[0].split('.')[0]
        
        image_desc = tokens[1:]
        
        if image_id in identifiers:
            if image_id not in descriptions.keys():
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' end_seq'
            
            descriptions[image_id].append(desc)
    
    return descriptions
    
def load_photo_features(filename,identifiers):
    
    all_features = load(open(filename,'rb'))
    features = {k:all_features[k] for k in identifiers}
    return features
    
def list_desc(descriptions):
    all_desc_list = list()
    for key in descriptions.keys():
        [all_desc_list.append(desc) for desc in descriptions[key] ]
    
    return all_desc_list
        
def toeknizer(descriptions):
    all_desc_list = list_desc(descriptions)
    tokens = Tokenizer()
    tokens.fit_on_texts(all_desc_list)
    return tokens
    
    
def max_length(descriptions):
    lines = list_desc(descriptions)
    return max(len(d.split()) for d in lines)
        
def input_sequences(tokenizer,max_length,text_train_descriptions,photos_train_features,vocab_Size):
    X1 , X2 , y = list(),list(),list()
    for key,desc_list in text_train_descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
                in_seq,out_seq = seq[:i],seq[i]
                in_seq = pad_sequences([in_seq],maxlen = max_length)[0]
                out_seq = to_categorical([out_seq],num_classes = vocab_Size)[0]
                #print (in_seq , '::',out_seq)
                X1.append(photos_train_features[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1),np.array(X2),np.array(y)

def progressive_input_sequences(tokenizer,max_length,desc,photos_train_features,vocab_Size):
    X1 , X2 , y = list(),list(),list()
    seq = tokenizer.texts_to_sequences([desc])
    for i in range(1,len(seq)):
        in_seq,out_seq = seq[:i],seq[i]
        in_seq = pad_sequences([in_seq],maxlen = max_length)[0]
        out_seq = to_categorical([out_seq],num_classes = vocab_Size)[0]
        #print (in_seq , '::',out_seq)
        X1.append(photos_train_features)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1),np.array(X2),np.array(y)

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
    #plot_model(model,to_file = 'model.png',show_shapes = True)
    
    return model

def progressive_learn(train_description,tokenzier,max_length,train_features_image):
    while 1:
        for Key,image in train_features_image.items():
            desc = train_description[Key]
            for des in desc:
                in_img , in_seq , out_Word = progressive_input_sequences(tokenizer,max_length,des,image,vocab_Size)
                yield[[in_img,in_seq],out_Word]
            
    
    
filename = 'Flickr8k_text/Flickr8k.token.txt'
print ("Reading the text description")

map_image_description = load_doc(filename)
clean_description(map_image_description)
vocab = make_vocabulary(map_image_description)
print ("Vocabulary Size : " ,len(vocab))
Description = "image_description.txt"
save_description(map_image_description,Description)


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = identifier_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = sequence_dict_description('image_description.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

tokenizer = toeknizer(train_descriptions)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
vocab_Size = len(tokenizer.word_index) + 1

print('Vocab Size %d' % vocab_Size)
max_length1 = max_length(train_descriptions)

#X1 , X2 , y = input_sequences(tokenizer,max_length1,train_descriptions,train_features,vocab_Size)


#load test dataset (1k)

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = identifier_set(filename)
print('Dataset: %d' % len(train))
# descriptions
test_descriptions = sequence_dict_description('image_description.txt', test)
print('Descriptions: test=%d' % len(train_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))


#X1_test , X2_test , y_test = input_sequences(tokenizer,max_length(test_descriptions),test_descriptions,test_features)



# define the model
model = build_model(vocab_Size, max_length(train_descriptions))
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
#model.fit([X1, X2], y, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1_test , X2_test], y_test))

print ("Creating Generator function")


model.fit_generator(progressive_learn(train_descriptions, tokenizer, max_length,train_features), steps_per_epoch=70000, verbose = 2 , callbacks = [checkpoint])
            
            
           
            
            
        
        
            
    
    
            
    
