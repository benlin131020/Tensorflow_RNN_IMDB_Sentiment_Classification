
#%%
import urllib.request
import os
import tarfile


#%%
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)


#%%
if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')

#%% [markdown]
# # 1. Import Library

#%%
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#%% [markdown]
# # 資料準備
#%% [markdown]
# # 讀取檔案

#%%
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


#%%
import os
def read_files(filetype):
    path = "data/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
    
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels,all_texts


#%%
y_train,train_text=read_files("train")


#%%
y_test,test_text=read_files("test")

#%% [markdown]
# # 查看正面評價的影評

#%%
train_text[0]


#%%
y_train[0]

#%% [markdown]
# # 查看負面評價的影評

#%%
train_text[12499]


#%%
y_train[12499]

#%% [markdown]
# # 先讀取所有文章建立字典，限制字典的數量為nb_words=2000

#%%
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)

#%% [markdown]
# # Tokenizer屬性
#%% [markdown]
# # fit_on_texts 讀取多少文章

#%%
print(token.document_count)


#%%
print(token.word_index)

#%% [markdown]
# # 將每一篇文章的文字轉換一連串的數字
# # 只有在字典中的文字會轉換為數字

#%%
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


#%%
print(train_text[0])


#%%
print(x_train_seq[0])

#%% [markdown]
# # 讓轉換後的數字長度相同
#%% [markdown]
# # 文章內的文字，轉換為數字後，每一篇的文章地所產生的數字長度都不同，因為後需要進行類神經網路的訓練，所以每一篇文章所產生的數字長度必須相同
# # 以下列程式碼為例maxlen=100，所以每一篇文章轉換為數字都必須為100

#%%
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)

#%% [markdown]
# # 如果文章轉成數字大於0,pad_sequences處理後，會truncate前面的數字

#%%
print('before pad_sequences length=',len(x_train_seq[0]))
print(x_train_seq[0])


#%%
print('after pad_sequences length=',len(x_train[0]))
print(x_train[0])

#%% [markdown]
# # 如果文章轉成數字不足100,pad_sequences處理後，前面會加上0

#%%
print('before pad_sequences length=',len(x_train_seq[1]))
print(x_train_seq[1])


#%%
print('after pad_sequences length=',len(x_train[1]))
print(x_train[1])

#%% [markdown]
# # 資料預處理

#%%
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)


#%%
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


#%%
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)


