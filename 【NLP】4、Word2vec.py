# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:36:13 2018

@author: wenzhe.jwz
"""
#####处理文件
import os 
import jieba 

os.chdir('D:/学习文档/机器学习')
f1 = open('Word2vec.txt',encoding = 'utf-8')
f2 = open('Word2vec_jieba.txt','a',encoding = 'utf-8')
lines = f1.readlines()
for line in lines:
    line.replace('\t','').replace('\n','').replace(' ','')
    seg_list = jieba.cut(line,cut_all = False,HMM = True)
    f2.write(" ".join(seg_list))
    
f1.close
f2.close

#####训练模型
from gensim.models import word2vec
import logging

#主程序
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)  
sentences =word2vec.Text8Corpus(u"Word2vec_jieba.txt")  
model =word2vec.Word2Vec(sentences, size=50)  #训练skip-gram模型，默认window=5  
print (model)

#1、计算两个词的相似度/相关程度  
try:  
    y1 = model.similarity("阿里", "万达")  
except:  
    y1 = 0  
print ("【国家】和【国务院】的相似度为：%s" % y1)  
print ("-----\n")  

#2、计算某个词的相关词列表
y2 = model.most_similar("阿里", topn=30)  # 20个最相关的  
print ("【阿里】最相关的词有:\r\n")
for item in y2:  
    print (item[0], item[1])  
print ("-----\n")  

#3、寻找对应关系  
print ("阿里巴巴-文娱，万达-")  
y3 =model.most_similar(['万达', '文娱'], ['阿里巴巴'], topn=3)  
for item in y3:  
    print (item[0], item[1])  
print ("----\n")  

#4、寻找不合群的词  
y4 =model.doesnt_match("阿里 文娱 内容".split())  
print ("不合群的词：", y4)  
print ("-----\n")  

# 保存模型，以便重用  
model.save("Word2vec.model")  
# 对应的加载方式  
model_2 =word2vec.Word2Vec.load("Word2vec.model")  
   
# 以一种c语言可以解析的形式存储词向量  
#model.save_word2vec_format(u"书评.model.bin", binary=True)  
# 对应的加载方式  
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)  



























