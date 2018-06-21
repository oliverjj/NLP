# -*- coding: utf-8 -*-  
import scipy as sp  
import numpy as np  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve 
from sklearn.metrics import classification_report  

'''''加载数据集，切分数据集80%训练，20%测试'''  
class tdidf(object):
    """
    步骤一：利用tfidf解析特征和label
    步骤二：利用朴素贝叶斯分类
    """    
    def __init__(self,datas):
        
        self.movie_reviews = load_files(datas)   #数据需要分析的文件夹
        #doc_terms_train, doc_terms_test, y_train, y_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.3)  
        '''''BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口'''  
        self.count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')  #CountVectorizer对应词频权重或是BOOL型权重(通过参数binary调节)向量空间模型
        self.count_vec.get_stop_words() #使用count_vec.get_stop_words()查看TfidfVectorizer内置的所有停用词

        self.a = self.count_vec.fit_transform(self.movie_reviews.data)  
        self.count_vec.get_feature_names()  #x按照count_vec分词结果
        
        self.x = self.a.toarray()
        self.y = self.movie_reviews.target 
        print(self.x) #tf-idf矩阵
        
    def train(self):
        #加载数据集，切分数据集80%训练，20%测试  
        z = 10
        the_all = 0
        lists = []
        for i in range(0,z):
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2,random_state = i)
            #调用MultinomialNB分类器  
            clf = MultinomialNB().fit(x_train, y_train) #clf:首先引入模块；其次将模块fit到训练样本中
            doc_class_predicted = clf.predict(x_test)  
            yy = round(float(np.mean(doc_class_predicted == y_test)),2)
            lists.append(yy)
        return lists

a = 'endata'
result = tdidf(a)
result.train()
