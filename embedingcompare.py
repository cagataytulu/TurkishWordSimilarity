import pandas as pd

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.fasttext import load_facebook_model

def runModel(model,dataSet):
	simPredict=[]
	oovPairCount=0
	oovList=[]
	for index, row in dataSet.iterrows():
	    if row.word1 in model.key_to_index and row.word2 in model.key_to_index:
	    	sim = model.similarity(row.word1, row.word2)
	    	simPredict.append(sim)
	    else:
	    	if row.word1 not in model.key_to_index and row.word1 not in oovList:
	    		oovList.append(row.word1)
	    	if row.word2 not in model.key_to_index and row.word2 not in oovList:
	    		oovList.append(row.word2)
	    	simPredict.append(None)
	    	oovPairCount+=1

	dataSet["simPredict"]=simPredict
	dataSet=dataSet.dropna()
	column_1 = dataSet["sim"]
	column_2 = dataSet["simPredict"]
	correlation = column_1.corr(column_2,method='spearman')
	resultDict={}
	resultDict["correlation"]=correlation
	resultDict["oovList"]=oovList
	resultDict["oovPairCount"]=oovPairCount
	resultDict["resultDataFrame"]=dataSet
	return resultDict
def getUniqueWordList(dataSet):
	wordsList=[]
	for index, row in dataSet.iterrows():
		if row.word1 not in wordsList:
			wordsList.append(row.word1)
		if row.word2 not in wordsList:
			wordsList.append(row.word2)
	return wordsList




df_turksim = pd.read_csv('data/turkishsim.txt', delimiter = ",",encoding="utf-8")
df_turksim["sim"]=df_turksim["sim"]/10
wList=getUniqueWordList(df_turksim)
print(f"Unique Word Count in SimTurk Dataset:{len(wList)}")
print(f"Unique Words in SimTurk Dataset:{wList}")

df_rg65 = pd.read_csv('data/rg65_eng_trk.txt', delimiter = ",",encoding="utf-8")
df_rg65["simEng"]=df_rg65["simEng"]/4
df_rg65["sim"]=df_rg65["sim"]/4
wList=getUniqueWordList(df_rg65)
print(f"Unique Word Count in RG65_Turkce Dataset:{len(wList)}")
print(f"Unique Words in RG65_Turkce Dataset:{wList}")

exit()

w2vModel= KeyedVectors.load_word2vec_format('models/trmodelwv', binary=True)
resultSet=runModel(w2vModel,df_turksim)
resultSet["resultDataFrame"].to_csv('data/result_SimTurk_w2v.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of SimTurk Dataset For Word2Vec Model:{resultSet['correlation']}")
print(f"OOV Pairs count for SimTurk Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for SimTurk Dataset:{resultSet['oovList']}")
print(f"Similarity result for SimTurk dataset for Word2Vec Model is Saved into data/result_SimTurk_w2v.csv")
print("==========================")

resultSet=runModel(w2vModel,df_rg65)
resultSet["resultDataFrame"].to_csv('data/result_Rg65_w2v.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of RG65_Turkce Dataset For Word2Vec Model:{resultSet['correlation']}")
print(f"OOV Pairs count for RG65 Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for RG65 Dataset:{resultSet['oovList']}")
print(f"Similarity result for RG65 dataset for Word2Vec Model is Saved into data/result_Rg65_w2v.csv\n\n")

#It might take 5-10 min to load fasttext model
fastTextModel=load_facebook_model('models/cc.tr.300.bin.gz').wv
resultSet=runModel(fastTextModel,df_turksim)
resultSet["resultDataFrame"].to_csv('data/result_SimTurk_fasttext.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of SimTurk Dataset For FastText Model:{resultSet['correlation']}")
print(f"OOV Pairs count for SimTurk Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for SimTurk Dataset:{resultSet['oovList']}")
print(f"Similarity result for SimTurk dataset for FastText Model is Saved into data/result_SimTurk_fasttext.csv")
print("==========================")

resultSet=runModel(fastTextModel,df_rg65)
resultSet["resultDataFrame"].to_csv('data/result_Rg65_Fasttext.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of RG65_Turkce Dataset For FastText Model:{resultSet['correlation']}")
print(f"OOV Pairs count for RG65 Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for RG65 Dataset:{resultSet['oovList']}")
print(f"Similarity result for RG65 dataset for FastText Model is Saved into data/result_Rg65_fasttext.csv\n\n")

glove_file = 'models/vectorsglovetr.txt'
glove_vec_file = 'models/glove_vec_file.txt'
glove2word2vec(glove_file, glove_vec_file)
gloveModel = KeyedVectors.load_word2vec_format(glove_vec_file)
resultSet=runModel(gloveModel,df_turksim)
resultSet["resultDataFrame"].to_csv('data/result_SimTurk_Glove.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of SimTurk Dataset For Glove Model:{resultSet['correlation']}")
print(f"OOV Pairs count for SimTurk Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for SimTurk Dataset:{resultSet['oovList']}")
print(f"Similarity result for SimTurk dataset for Glove Model is Saved into data/result_SimTurk_glove.csv")
print("==========================")

resultSet=runModel(gloveModel,df_rg65)
resultSet["resultDataFrame"].to_csv('data/result_Rg65_Glove.csv',encoding='utf-8')
print(f"Spearman Rank Correlation Value of RG65_Turkce Dataset For Glove Model:{resultSet['correlation']}")
print(f"OOV Pairs count for RG65 Dataset:{resultSet['oovPairCount']}")
print(f"OOV Count:{len(resultSet['oovList'])} and OOV List for RG65 Dataset:{resultSet['oovList']}")
print(f"Similarity result for RG65 dataset for Glove Model is Saved into data/result_Rg65_glove.csv")

