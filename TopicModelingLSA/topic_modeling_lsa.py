# topic_modeling_lsa.py
# Extract topics from texts using latent semantic anaylsis.
# Source: https://towardsdatascience.com/topic-modeling-with-latent-
# semantic-analysis-58aeab6ab2f2
# Source (dataset): https://www.kaggle.com/datasets/eswarchandt/amazon-
# music-reviews?select=Musical_instruments_reviews.csv
# Windows/MacOS/Linux
# Python 3.7


import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_short, stem_text
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel


def main():
	# Introduction
	# Topic modelling is an unsupervised learning approach that allows
	# for a way to extract topics from documents. It plays a vital role
	# in many applications such as document clustering and information
	# retrieval. This example will provide an overview of one of the
	# most popular methods of topic modeling: Latent Semantic Analysis.
	# Note, in NLP, a "topic" is defined by a collection of words that
	# are strongly associated. Since documents are not restricted by a
	# limited set of words, they usually contain multiple topics. A 
	# document can be assigned to a topic by finding the topic that the
	# document is most strongly associated with.

	# Latent Semantic Analysis
	# Latent Semantic Analysis (LSA) is a method that allows us to
	# extract topics from documents by converting their text into word-
	# topic and document-topic matrices. The procedure for LSA is
	# relatively straightforward:
	# 1) convert the text corpus into a document-term matrix.
	#	-> Before deriving topics from documents, the text has to be
	#		converted into a document-term matrix. This is often done
	#		with the bag of words or TF-IDF algorithm.
	# 2) implment truncated singular value decomposition (SVD)
	#	-> Truncated singular value decomposition is the heart of LSA.
	#		The operation is key to obtaining topics from the given
	#		collection of documents. The formula is 
	#		Anxm = UnxrSrxr(V^T)mxr. In layman's terms, the operation
	#		decomposes the high dimensional document-term matrix into 3
	#		smaller matrices (U, S, and V).
	#	-> The variable A represents the document-term matrix, with a
	#		count-based value assigned between each document and word
	#		pairing. The matrix has n x m dimensions, with n 
	#		representing the number of documents and m representing the
	#		number of words.
	#	-> The variable U representsi the document-topic matrix.
	#		Essentially, its values show the strength of association 
	#		between each document and its derived topics. The matrix
	#		has n x r dimensions, with n representing the number of
	#		documents and r representing the number of topics.
	#	-> The variable S represents a diagonal matrix that evaluates
	#		the "strength" of each topic in the collection of 
	#		documents. The matrix has r x r dimensions, with r 
	#		representing the number of topics.
	#	-> The variable V represents the word-topic matrix. Its values
	#		show the string the each association between each word and
	#		the derived topics. The matrix has m x r dimensions, with
	#		m representing the number of words and r representing the
	#		number of topics.
	#	-> Note that while the number of documents and words in a 
	#		corpus is always constant, the number of topics is not a
	#		fixed variable as it is decided by the ones who run the
	#		operation. As a result, the output of an SVD depends on the
	#		number of topics you wish to extract.
	# 3) encode the words/documents with the extracted topics.
	#	-> With the SVD operations, we can convert the document-term
	#		matrix into a document-topic matrix (U) and a word-topic
	#		matrix (V). These matrices allow us to find the words with
	#		the strongest association with each topic. This information
	#		can be used to decide what each derived topic represents as
	#		well as determine which documents belong to which topics.

	# Limitations
	# LSA enables us to uncover the underlying topics in documents with
	# speed and efficiency, however it does have its drawbacks. For
	# starters, some information loss is inevitable when conducting
	# LSA. When docments are converted into a document-term matrix,
	# word order is completely neglected. Since word order plays a big
	# role in the semantic value of words, omitting it leads to
	# information loss during the topic modeling process. 
	# Furthermore, LSA is unable to account for homonymy or polysymy.
	# Since the technique evaluates words based on the context they are
	# presented in, it is unable to identify words with multiple
	# meanings and distinguish these words by their use in the text.
	# It is also difficult to determin the optimal number of topics for
	# a give set of documents. While there are several schools of
	# thought with regards to finding the ideal number of topics to
	# represent a collection of documents, there isn't a sure-fire
	# approach towards achieving this.
	# Finally, LSA lacks interpretability. Even after successfully
	# extracting topics with sets of words with strong associations, it
	# can be challenging to draw insights from them since it is
	# difficult to determine what topic each set of terms represents.

	# Case Study/Example
	# This exmaple will primarily utilize the gensim library, an open-
	# source library that specializes in topic modeling. The dataset
	# will be a CSV file containing the reviews of musical instruments
	# and can be obtained (copyright-free) from kaggle.

	# Load and preview the data.
	df = pd.read_csv(
		"Musical_instruments_reviews.csv",
		usecols=["reviewerID", "reviewText"]
	).dropna() # drop NaN from dataset (yes, they are there)
	print(df.head())


	# The first step is to convert these reviews into a document-term
	# matrix. For that, perform som preprocessing on the text (remove
	# punctuation, lowercase the text, remove stop words, remove short
	# words, and reduce every word to is base form with stemming).
	def preprocess(text):
		CUSTOM_FILTERS = [
			lambda x: x.lower(),
			remove_stopwords,
			strip_punctuation,
			strip_short,
			stem_text
		]
		text = preprocess_string(text, CUSTOM_FILTERS)

		return text


	# Apply preprocessing function to reviews.
	df["Text (Clean)"] = df["reviewText"].apply(
		lambda x: preprocess(x)
	)
	print(df.head())

	# Convert these procesed reviews into a document-term matrix with a
	# bag of words model.
	corpus = df["Text (Clean)"]
	dictionary = corpora.Dictionary(corpus)

	bow = [dictionary.doc2bow(text) for text in corpus]

	# Determine the number of topics that should be extracted from the
	# reviews. One approach towards finding the best number of topics
	# is using the coherence score metric. The coherence score
	# essentially shows how similar the words from each topic are in
	# terms of semantic value, with a higher score corresponding to 
	# higher similarity.
	for i in range(2, 11):
		lsi = LsiModel(bow, num_topics=i, id2word=dictionary)
		coherence_model = CoherenceModel(
			model=lsi, texts=df["Text (Clean)"], dictionary=dictionary,
			coherence="c_v"
		)
		coherence_score = coherence_model.get_coherence()
		print(
			"Coherence score with {} clusters: {}".format(
				i, coherence_score
			)
		)

	# We see that the coherence score is highest with 2 topics, so that
	# is the number of topics we will extract when performing SVD.

	# Implement the truncated single value decomposition on this
	# matrix. use the LSImodel from the gensim library to build a model
	# that performs SVD on the given matrix.
	lsi = LsiModel(bow, num_topics=2, id2word=dictionary)

	# We are able to obtain 2 topics from the document-term matrix. As
	# a result, we can see which words have the strongest association
	# with each topic and infer what these topics represent. Let's see
	# the 5 words that each topic has the strongest association to.
	for topic_num, words in lsi.print_topics(num_words=5):
		print("Words in {}: {}.".format(topic_num, words))

	# Based on the given words, topic 0 may represent reviews that
	# address the sound or noise that is made when using the product,
	# while topic 1 may represent reviews that address the pieces of
	# equipment themselves. Additionally, we can see what values the
	# model assigns for every document and topic pairing.
	# As previously mentioned, documents usually have multiple topics.
	# However, some topics have a stronger association with the
	# documents than others. So, we can determine which topic a
	# document belongs to by finding the one that registers the highest
	# value by magnitude.

	# Find the scores given between the review and each topic.
	corpus_lsi = lsi[bow]
	score1 = []
	score2 = []
	for doc in corpus_lsi:
		score1.append(round(doc[0][1], 2))
		score2.append(round(doc[1][1], 2))

	# Create a dataframe that shows scores assigned for both topics for
	# each review.
	df_topic = pd.DataFrame()
	df_topic["Text"] = df["reviewText"]
	df_topic["Topic 0 score"] = score1
	df_topic["Topic 1 score"] = score2
	df_topic["Topic"] = df_topic[
		["Topic 0 score", "Topic 1 score"]
	].apply(lambda x: x.values.argmax(), axis=1)
	print(df_topic.head(1))

	# Find a sample from each topic.
	df_topic0 = df_topic[df_topic["Topic"] == 0]
	df_topic1 = df_topic[df_topic["Topic"] == 1]
	print("Sample text from topic 0:\n{}\n".format(
		df_topic0.sample(1, random_state=2)["Text"].values
	))
	print("Sample text from topic 1:\n{}\n".format(
		df_topic1.sample(1, random_state=2)["Text"].values
	))

	# Sample text from topic 0 discusses the sound of their instrument
	# after buying the tube screamer, whereas the sample text from
	# topic 1 focuses more on the quality of the purchased pedal
	# itself. This is in line with the interpretation of the two
	# derived topics.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()