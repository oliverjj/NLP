# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:00:07 2018

@author: wenzhe.jwz
"""
#LDA算法    
print 'LDA:'    
import numpy as np    
import lda    
import lda.datasets    
model = lda.LDA(n_topics=2, n_iter=500, random_state=1)    
model.fit(np.asarray(weight))     # model.fit_transform(X) is also available    
topic_word = model.topic_word_    # model.components_ also works  
  
#输出主题中的TopN关键词  
word = vectorizer.get_feature_names()  
for w in word:  
    print w  
print topic_word[:, :3]  
n = 5    
for i, topic_dist in enumerate(topic_word):    
    topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]    
    print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))    
  
  
#文档-主题（Document-Topic）分布    
doc_topic = model.doc_topic_    
print("type(doc_topic): {}".format(type(doc_topic)))    
print("shape: {}".format(doc_topic.shape))   