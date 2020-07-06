import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter

class basic_lda():
    def __init__(self):
        self.data = np.load('lda.npz')
        self.data_train = self.data['train']
        self.data_test = self.data['test']

        self.data_train = self.data_train[self.data_train[:,0]<200]
        self.data_test = self.data_train[self.data_train[:,0]<50]

        self.data_train[:,0] = self.data_train[:,0]-1
        self.data_test[:,0] = self.data_test[:,0]-1
        self.data_train[:,1] = self.data_train[:,1]-1
        self.data_test[:,1] = self.data_test[:,1]-1

        self.voc_num = max(self.data_train[:,1])+1
        self.doc_num = max(self.data_train[:,0])+1

        print ('The number of voc is {}'.format(self.voc_num))
        print ('The number of doc is {}'.format(self.doc_num))


        #hyper_parameter
        ### doc-topic
        self.topic_num = 10
        self.alpha = 3
        ### topic-word
        self.beta = 3
        self.result =[]
        self.training_times = 0

    def model_init(self):
        print ('Initilization starting')
        #变成word_num都是1
        self.res = []
        for vec in self.data_train:
            if vec[-1] ==1:
                self.res.append(vec)
            else:
                new_vec = vec.copy()
                new_vec[-1]=1
                for _ in range(vec[-1]):
                    self.res.append(new_vec)


        #初始化
        self.doc_word_sum = np.zeros(self.doc_num)
        self.topic_word_sum = np.zeros(self.topic_num)
        self.doc_topic_count = np.zeros((self.doc_num,self.topic_num))
        self.word_topic_count = np.zeros((self.voc_num,self.topic_num))
        self.topic_att = []
        for doc_idx,word_idx,word_num in self.res:
            topic = np.random.randint(0,self.topic_num,1)[0] #随机初始化主题
            self.topic_att.append(topic)
            
            self.doc_word_sum[doc_idx]+=1 #文档数量加上单词数量
            self.topic_word_sum[topic]+=1 
            
            self.doc_topic_count[doc_idx,topic]+=1
            self.word_topic_count[word_idx,topic]+=1
    
    def gibbs_sampling(self):
        nt = []
        for ((doc_idx,word_idx,word_num),topic) in zip(self.res,self.topic_att):           
            self.doc_topic_count[doc_idx,topic]-=1 #词在各topic的数量
            self.word_topic_count[word_idx,topic]-=1 #每个doc中topic的总数
            self.doc_word_sum[doc_idx]-=1 #nwsum 每个topic词的总数
            self.topic_word_sum[topic]-=1 #每个doc中词的总数

            topic_doc =  (self.doc_topic_count[doc_idx]+self.alpha)/(self.doc_word_sum[doc_idx]
                            + self.topic_num*self.alpha) #

            word_topic = (self.word_topic_count[word_idx]+self.beta)/(self.topic_word_sum+ 
                            self.voc_num*self.beta )

            prob = topic_doc*word_topic
            prob = prob/sum(prob)

            new_topic = np.random.multinomial(1,prob,size=1)[0].argmax()

            self.doc_topic_count[doc_idx,new_topic]+=1 #词在各topic的数量
            self.word_topic_count[word_idx,new_topic]+=1 #每个doc中topic的总数
            self.doc_word_sum[doc_idx]+=1 #nwsum 每个topic词的总数
            self.topic_word_sum[new_topic]+=1 #每个doc中词的总数
        
            nt.append(new_topic)
        
        self.topic_att = nt

    def complexity(self):
        print ('Starting cal complexity')
        error_sum= 0
        doc_pb = (self.doc_topic_count + self.alpha) / (self.doc_word_sum + self.topic_num*self.alpha).reshape(-1,1)[0]
        word_pb = (self.word_topic_count + self.beta) / (self.topic_word_sum +self.voc_num*self.beta)
        for doc_idx,word_idx,word_num in self.res:
            error_sum+=np.log(np.sum(doc_pb[doc_idx] * word_pb[word_idx]))
        loss = np.exp(-error_sum/len(self.res))
        print ('adsfasdfadsf',loss)
        return loss

    def main_fun(self):
        for i in range(300):
            
            self.training_times+=1
            print ('starting {}th trainging'.format(self.training_times))
            self.gibbs_sampling()
            loss = self.complexity()
            self.result.append(loss)
            print ('The loss of training set is .....{}'.format(loss))



if __name__ == "__main__":
    res = basic_lda()
    res.model_init()
    res.main_fun()


