import urllib.request
import os
import tarfile
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer

NUM_CLASSES = 2
SEQ_MAX_LEN = 100

class IMDBData():
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.seqlen_train = []
        self.seqlen_test = []
        self.seq_max_len = 0
        self.batch_id = 0
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        filepath = "data/aclImdb_v1.tar.gz"
        if not os.path.isfile(filepath):
            result = urllib.request.urlretrieve(url, filepath)
            print('downloaded:', result)
        if not os.path.exists("data/aclImdb"):
            tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
            result = tfile.extractall('data/')
        self.y_train, train_text = self.read_files("train")
        self.y_test, test_text = self.read_files("test")
        print("preprocessing...")
        #先讀取所有文章建立字典，限制字典的數量為nb_words=2000
        token = Tokenizer(num_words=2000)
        token.fit_on_texts(train_text)
        #將每一篇文章的文字轉換一連串的數字
        #只有在字典中的文字會轉換為數字
        x_train_seq = token.texts_to_sequences(train_text)
        x_test_seq  = token.texts_to_sequences(test_text)
        for seq in x_train_seq:
            self.seqlen_train.append(len(seq))
        for seq in x_test_seq:
            self.seqlen_test.append(len(seq))
        #文章內的文字，轉換為數字後，每一篇的文章地所產生的數字長度都不同
        #因為後需要進行類神經網路的訓練，所以每一篇文章所產生的數字長度必須相同
        #以下列程式碼為例maxlen=100，所以每一篇文章轉換為數字都必須為100
        #max_len_train = max(map(len, x_train_seq))
        #max_len_test = max(map(len, x_test_seq))
        #self.seq_max_len = max(max_len_train, max_len_test)
        self.seq_max_len = SEQ_MAX_LEN
        x_train_pad = sequence.pad_sequences(x_train_seq, maxlen=self.seq_max_len)
        x_test_pad = sequence.pad_sequences(x_test_seq, maxlen=self.seq_max_len)
        self.x_train = np.reshape(x_train_pad, [-1, self.seq_max_len, 1])
        self.x_test = np.reshape(x_test_pad, [-1, self.seq_max_len, 1])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.y_train = sess.run(tf.one_hot(self.y_train, NUM_CLASSES))
            self.y_test = sess.run(tf.one_hot(self.y_test, NUM_CLASSES))

    def rm_tags(self, text):
        re_tag = re.compile(r'<[^>]+>')
        return re_tag.sub('', text)

    def read_files(self, filetype):
        path = "data/aclImdb/"
        file_list = []
        positive_path = path + filetype+"/pos/"
        for f in os.listdir(positive_path):
            file_list += [positive_path+f]
        negative_path = path + filetype+"/neg/"
        for f in os.listdir(negative_path):
            file_list += [negative_path+f]
        print('read', filetype, 'files:', len(file_list))
        all_labels = ([1] * 12500 + [0] * 12500) 
        all_texts = []
        for fi in file_list:
            with open(fi, encoding='utf8') as file_input:
                all_texts += [self.rm_tags(" ".join(file_input.readlines()))]
                
        return all_labels, all_texts

    def get_all(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test), (self.seqlen_train, self.seqlen_test), (self.seq_max_len)

    def next_batch(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.x_train):
            self.batch_id = 0
        batch_data = (self.x_train[self.batch_id:min(self.batch_id + batch_size, len(self.x_train))])
        batch_labels = (self.y_train[self.batch_id:min(self.batch_id + batch_size, len(self.x_train))])
        batch_seqlen = (self.seqlen_train[self.batch_id:min(self.batch_id + batch_size, len(self.x_train))])
        self.batch_id = min(self.batch_id + batch_size, len(self.x_train))
        return batch_data, batch_labels, batch_seqlen
