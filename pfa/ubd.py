import pandas as pd
import numpy as np

class layer_para():
    def __init__(self,layer):
        self.layer_len = layer
        self.topic_num = 10
        self.word_num = 100
        
        
        self.hyper_m = 200
        self.hyper_alphaw = 3
        self.hyper_alphad = 3
        
        self.layer_info = []
        
        self.initlize(layer)
            
    def initlize(self,layer):
        
        for  layer_num in range(layer):
            if layer_num == 0:
                tw = np.random.dirichlet([3]*self.word_num,self.topic_num).T
                
                m = np.random.poisson(lam = self.hyper_m)

                wi = np.random.dirichlet([self.hyper_alphaw]*self.topic_num)

                node_word_node = np.zeros((self.topic_num,self.word_num*self.topic_num),dtype=int)


                for topic_idx in range(self.topic_num):
                    piw = np.kron(tw[:,topic_idx],wi)
                    node_word_node[topic_idx] = np.random.multinomial(n = self.hyper_m, pvals = piw)

                word_topic_count = np.sum(node_word_node,0)
                word_topic_count = word_topic_count.reshape(self.word_num,self.topic_num)

                ad = np.array([self.hyper_alphad] * 100)
        
                self.layer_info.append({'m':m,'wi':wi,'word_topic_count':word_topic_count,'node_word_node':node_word_node})
                
                tw = np.zeros((self.word_num,self.topic_num))
                for i in range(self.topic_num):
                    tw[:,i] = np.random.dirichlet(ad+word_topic_count[:,i])
            else:
                m = np.random.poisson(lam = self.hyper_m)

                wi = np.random.dirichlet([self.hyper_alphaw]*self.topic_num)

                node_word_node = np.zeros((self.topic_num,self.word_num*self.topic_num),dtype=int)


                for topic_idx in range(self.topic_num):
                    piw = np.kron(tw[:,topic_idx],wi)
                    node_word_node[topic_idx] = np.random.multinomial(n = self.hyper_m, pvals = piw)

                word_topic_count = np.sum(node_word_node,0)
                word_topic_count = word_topic_count.reshape(self.word_num,self.topic_num)

                ad = np.array([self.hyper_alphad] * 100)
        
                self.layer_info.append({'m':m,'wi':wi,'word_topic_count':word_topic_count,'node_word_node':node_word_node})
                
                
                tw = np.zeros((self.word_num,self.topic_num))
                for i in range(self.topic_num):
                    tw[:,i] = np.random.dirichlet(ad+word_topic_count[:,i])