# TurkishWordSimilarity
This project is intented to share the codes and results of Turkish Word Similarity Tests conducted using Turkish embeding vectors creatred with Word2Vec, Glove and FastText methods.

#requirements
gensim librarry and 
model files of Turkish Word2Vec, Glove and FastText word embeddings

To get the Word2Vec Turkish model file, please refer to the https://github.com/akoksal/Turkish-Word2Vec
To get the Glove Turkish model files please refer to the https://github.com/inzva/Turkish-GloVe ## Please download vectors.bin.gz vectors.txt.gz files and unzip them
To get the model file of FastText Turkish https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.bin.gz

After you have donwloaded the model files, please put them into the models directory

then run the embedingcompare.py python file like;
python embedingcompare.py

you will get the results of the similarity tests of SimTurk and RG65_Turkce datasets for each model under the results directory.
