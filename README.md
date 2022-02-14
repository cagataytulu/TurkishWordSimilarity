# TurkishWordSimilarity
This project is intented to share the codes and results of Turkish Word Similarity Tests conducted using pre-trained Turkish embeding vectors created with Word2Vec, Glove and FastText methods.

#requirements
gensim librarry and 
model files of Turkish Word2Vec, Glove and FastText word embeddings

To get the pre-trained Word2Vec Turkish model file, please refer to the https://github.com/akoksal/Turkish-Word2Vec
To get the pre-trained Glove Turkish model files please refer to the https://github.com/inzva/Turkish-GloVe ## Please download vectors.bin.gz vectors.txt.gz files and then unzip them
To get the pre-trained model file of FastText Turkish https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.bin.gz

First clone the codes into your local disk, 
Download the model files as explained above, please put them into the models directory

then run the embedingcompare.py python file like;
python embedingcompare.py

you will get the results of the similarity tests of SimTurk and RG65_Turkce datasets for each model under the results directory.
