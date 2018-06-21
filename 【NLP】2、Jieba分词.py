# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:37:13 2018

@author: wenzhe.jwz
"""
jieba.cut 方法接受三个输入参数: 需要分词的字符串；cut_all 参数用来控制是否采用全模式；HMM 参数用来控制是否使用 HMM 模型
jieba.cut_for_search 方法接受两个参数：需要分词的字符串；是否使用 HMM 模型。该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细
待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。注意：不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8
jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
jieba.Tokenizer(dictionary=DEFAULT_DICT) 新建自定义分词器，可用于同时使用不同词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。
############1、分词

import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut("他来到了网易杭研大厦",HMM=True)  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


print('/'.join(jieba.cut('如果放到post中将出错。', HMM=True)))

jieba.suggest_freq(('中', '将'), True)

print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))

jieba.suggest_freq('台中', True)

print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))

import jieba.analyse

jieba.analyse.extract_tags('我是一只小熊', topK=20, withWeight=True, allowPOS=())
jieba.analyse.TFIDF(idf_path="D:/学习文档/机器学习/endata/pos/nb!.txt")

import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))
    
##############2、例子
import jieba

test_sent = u"永和服装饰品有限公司"
result = jieba.tokenize(test_sent) ##Tokenize：返回词语在原文的起始位置
for tk in result:
    print(tk[0],tk[1],tk[2])
    print "word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2])
    print tk
    
    
    
    
    