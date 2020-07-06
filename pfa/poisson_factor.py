import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter

class basic_pfa():
    def __init__(self):
        self.data = np.load('lda.npz')
        self.data_train = self.data['train']
        self.data_test = self.data['test']

        self.data_train = self.data_train[self.data_train[:,0]<200]
        self.data_test = self.data_train[self.data_train[:,0]<200]

        self.data_train[:,0] = self.data_train[:,0]-1
        self.data_test[:,0] = self.data_test[:,0]-1
        self.data_train[:,1] = self.data_train[:,1]-1
        self.data_test[:,1] = self.data_test[:,1]-1

        self.voc_num = max(self.data_train[:,1])+1
        self.doc_num = max(self.data_train[:,0])+1

        print ('The number of voc is {}'.format(self.voc_num))
        print ('The number of doc is {}'.format(self.doc_num))
        

        #hyper-parameter
        self.eta = 0.5
        self.c = 1
        self.c0 = 1
        self.r0 = 1
        self.gamma = 1
        self.alpha = 0.05
        self.eps = 0.05
        
        #number of document,word type,topic
        self.topic_num = 10 #主题数量

         #gengerative process sample
        self.phi = sts.dirichlet.rvs([self.alpha]*self.voc_num,size = self.topic_num)  #Topic-word matrix  (voc * k matrix)
        self.pk = sts.beta.rvs(self.c*self.eps,self.c*(1-self.eps),size =self.topic_num ) # pk 的分布 1*k matrix
        self.rk = sts.gamma.rvs(self.c0*self.r0,scale=1/self.c0,size=self.topic_num ) #rk 1*k matrix
        self.ki = np.array([sts.gamma.rvs(self.rk,scale = self.pk/(1-self.pk)) for i in range(self.doc_num)]) #doc_num*k 


        #sampling count
        self.xik = np.zeros((self.doc_num,self.topic_num)) #第i个主题第k个topic
        self.xpk = np.zeros((self.voc_num,self.topic_num)) #第k个主题第p个词的数量
        self.xk = np.zeros((self.topic_num))           #第k个topic的数量
        self.xi = np.zeros((self.doc_num))         #第i个文档的数量

        self.per_plot = []

    def gibbs_sampling(self):
        print ('Sampling start')
        #1.Count assignment
        for doc_word_count in self.data_train:
            doc_index,word_index,word_count = doc_word_count #[0,7,2]
            
            prob =   self.phi[:,word_index] * self.ki[doc_index] # 单词在每个主题的概率 * 该文档每个主题的概率
            prob/=sum(prob+0.1/self.topic_num) 
            res = sts.multinomial.rvs(word_count,p=prob)
                
            self.xik[doc_index] += res   #第i个文档第k个topic的数量
            self.xpk[word_index] += res          #第k个主题第p个词的数量
            self.xk += res                       #第k个topic的数量
            self.xi[doc_index] += word_count     #第i个文档的数量

        #更新topic-doc  matrix,就是我们要替换的部分
        for topic_index in range(self.topic_num):
            self.phi[topic_index] = sts.dirichlet.rvs([self.alpha]*self.voc_num + self.xpk[:,topic_index])

         #更新pk    
        for topic_index in range(self.topic_num):   
            self.pk[topic_index] = sts.beta.rvs(self.c * self.eps + self.xk[topic_index] ,
                                    self.c *(1- self.eps)+ self.doc_num *  self.rk[topic_index] )

        #sample rk
        for topic_index in range(self.topic_num):    
            if self.xk[topic_index] == 0: #当xk=0,负二项分布退化为伯努利分布
                self.rk[topic_index] = sts.gamma.rvs(self.c0 * self.r0 , 
                                        scale = 1/(self.c0 - self.doc_num * np.log(1- self.pk[topic_index])))
            else:
                self.rk[topic_index] = sts.gamma.rvs(self.c0 * self.r0 + np.sum(self.CRT(topic_index)),
                                    scale = 1/(self.c0 - self.doc_num * np.log(1- self.pk[topic_index])))

        # sample self.ki
        #for topic_index in range(self.k): 
        self.ki = sts.gamma(self.xik + self.rk,self.pk).rvs()
    
    def CRT(self,topic): #CRT 
        res = np.array([sum([sts.bernoulli.rvs(self.pk[topic]/(i-1+self.pk[topic]))
         for i in np.linspace(1,self.xik[d][topic],int(self.xik[d][topic]))]) 
         for d in range(self.doc_num)])
        return res
    

    def complexity(self):
        print ('calculating loss start')
        self.phi_theta = np.array([np.sum(self.phi * self.ki[doc_index].reshape(-1,1),0) for doc_index in range(self.doc_num)])
        self.phi_theta = self.phi_theta/np.sum(self.phi_theta,1).reshape(-1,1)  
        error_sum = 0
        for doc_index,word_index,num in self.data_test:
            error_sum+= num*np.log(self.phi_theta[doc_index,word_index])
        print (error_sum)
        loss = np.exp(-error_sum/(np.sum(self.data_test[:,-1])))
        return loss

    
    def main_fun(self):
        for i in range(20):
            print ('starting {} th training'.format(i))
            self.gibbs_sampling()
            loss = self.complexity()
            self.per_plot.append(loss)
            print (loss)



if __name__ == "__main__":
    res = basic_pfa()
    res.main_fun()