# news_classifier.py
# Detect fake news with LSTM and BERT models.
# Source: https://towardsdatascience.com/the-fight-against-fake-news-
# with-deep-learning-6c41dd9eaae4
# Source (GitHub): https://github.com/shayaf84/Fake-News-Detection
# Source (Dataset): https://www.uvic.ca/ecs/ece/isot/datasets/fake-
# news/index.php
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import string
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# Download all necessary nltk modules.
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def main():
	# Exploratory Data Analysis
	# Because the real and fake news files are separate, we need to
	# label and concatenate the two data frames. Real news will be
	# labeled 0 and fake news 1.
	real = pd.read_csv("./News_dataset/True.csv")
	fake = pd.read_csv("./News_dataset/Fake.csv")

	# Shape of the real and fake news datasets.
	print("Real news: ", real.shape)
	print("Fake news: ", fake.shape)

	# Assign labels and append to dataframe.
	class0 = []
	for i in range(len(real)):
		class0.append(0)
	real.insert(4, "class", class0, True)

	class1 = []
	for i in range(len(fake)):
		class1.append(1)
	fake.insert(4, "class", class1, True)

	# Concatenate the fake and real news dataframes into one.
	total = pd.concat([real, fake])
	total = total.sample(frac=1)
	print(total.head())
	print(total.columns)

	# Note that there is approximately a 48:52 real:fake news ratio.
	# This means that the dataset is relatively balanced. For the
	# purposes of the project, we only want the title and class
	# columns.
	data = total[["title", "class"]]
	print(data.head())

	# Now that the dataset is cleaned, analyze trends found within it.
	# Analyze the mean, minimum, and maximum character length of the
	# titles. This frequency is plotted with a histogram.
	print("Mean Length,", data["title"].apply(len).mean())
	print("Min Length,", data["title"].apply(len).min())
	print("Max Length,", data["title"].apply(len).max())

	# Notice how that the number of characters in each entry ranges
	# from 8-256. There is a high concentration of samples with a
	# length of 50-100. This can be further seen with the mean length
	# in the dataset being approximately 80.
	# plt.hist(data["title"].apply(len))
	# plt.ylabel("frequency")
	# plt.xlabel("title character length")
	# plt.show()


	# Preprocess the Data
	# Conduct some initial preprocessing using the string module. This
	# includes lowercasing all characters and removing punctuation.

	# Lowercase letters.
	data["title"] = data["title"].str.lower()
	print(data.head())

	# Ensure that all necessary punctuation is in one list (include '
	# and " as they are not by default).
	punc = list(string.punctuation) + ["\'", "\""]
	print("Punctuation: ", punc)


	# Iterate through dataframe and remove all punctuation.
	def removePunc(text):
		for i in punc:
			text = text.replace(i, "")
		return text


	# Apply to dataframe.
	data["title"] = data["title"].apply(removePunc)
	print(data.head())

	# Use the NLTK library to conduct further preprocessing on the
	# dataset.
	# -> tokenization: splitting a text into a smaller unit called a
	#	token (each individual word will be an index in an array).
	# -> lemmatization: removing the word's inflectional endings. For
	#	example, the word "children" will be lemmatized to "child".
	# -> removal of stop words: commonly used words such as "the" or
	#	"for" will be removed, as they take up space in the dataset.
	data["title"] = data.apply(
		lambda row: nltk.word_tokenize(row["title"]),
		axis=1
	)

	lemmatizer = WordNetLemmatizer()


	def lemma(data):
		return [lemmatizer.lemmatize(w) for w in data]


	data["title"] = data["title"].apply(lemma)

	stop = stopwords.words("english")
	data["title"] = data["title"].apply(
		lambda x: [i for i in x if i not in stop]
	)
	print(data.head())

	# This project will construct two models to classify the text:
	# -> LSTM model (use Tensorflow's wiki-words-250 embeddings)
	# -> BERT model


	# Creating an LSTM Model
	# Split the dataset into an 80:20 train:test ratio.
	titles = data["title"].values
	labels = data["class"].values
	title_train, title_test, y_train, y_test = train_test_split(
		titles, labels, test_size=0.2, random_state=1000
	)

	# We need to convert the text into a vector format before it can be
	# processed by the model. Tensorflow's wiki-words-250 uses a
	# Word2Vec Skip-gram architecture (skip-gram is training by
	# predicting the context based on an input word).
	embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

	indiv = []
	for i in title_train:
		temp = np.array(embed(i))
		indiv.append(temp)

	# Account for different length of words
	indiv = tf.keras.preprocessing.sequence.pad_sequences(
		indiv, dtype="float"
	)
	print(indiv.shape) # (number of samples, max_lentgth, num_features)

	# Convert each of the testing data series to a Word2Vec embedding.
	test = []
	for i in title_test:
		temp = np.array(embed(i))
		test.append(temp)
	test = tf.keras.preprocessing.sequence.pad_sequences(
		test, dtype="float"
	)

	# Construct the model consisting of:
	# -> 1 LSTM layer with 50 units.
	# -> 2 Dense layers (one with 20 neurons and the other with 5),
	#	with a ReLU activation function.
	# -> 1 Dense output layer with a sigmoid activation function.
	# Use the Adam optimizer, binary crossentropy loss, and a
	# performance metric of accuracy to train the model. The model will
	# be trained over 10 epochs. These hyperparameters can be adjusted.
	model = tf.keras.models.Sequential(
		[
			tf.keras.layers.LSTM(50),
			tf.keras.layers.Dense(20, activation="relu"),
			tf.keras.layers.Dense(5, activation="relu"),
			tf.keras.layers.Dense(1, activation="sigmoid"),
		]
	)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)

	# This model will have a maximum accuracy of around 91.5% on the
	# test data.
	'''
	model.fit(
		indiv, y_train, validation_data=(test, y_test), epochs=10
	)
	'''


	# Introducing BERT
	# The problem with Word2Vec is that it generates the same embedding
	# regardles of the way in which the words is used. To combat
	# against this, use BERT, which can generate contextualized
	# embeddings.
	# BERT (Bidirectional Encoder Representations from Transformers)
	# takes advantage of attention mechanisms to generate
	# contextualized embeddings.
	# A transformer model uses an encoder-decoder architecture. The
	# encoder layer generates a continuous representation which
	# consists of the information learned from the input. The decoder
	# layer creates an output, with the previous input being passed
	# into the model. BERT only uses an encoder as its goal is to
	# generate a vector representation to gain information from the
	# text.


	# Pre-training and Fine-tuning BERT
	# There are two methods to train BERT. The first is called masked
	# language modelling. Before passing sequences, 15% of the words
	# are replaced with a [MASK] token. The model will predict the
	# masked words, using the context provided by the unmasked ones.
	# This is done by
	# -> applying a classification layer on the encoder output,
	#	consisting of an embedding matrix. Hence, it will be the same
	#	size as that of the vocabulary.
	# -> calculating the probability of the word with the softmax
	#	function.
	# The second method is next sentence prediction. The model will
	# recieve two sentences as input, and predict a binary value of
	# whether the second sentence follows the first. While training,
	# 50% of inputs are pairs, while the other 50% are random sentences
	# from the corpus. To differentiate between the two sentences
	# -> a [CLS] token is added at the beginning of the first sentence,
	#	and an [SEP] token is added at the end of each.
	# -> each token (word) has a positional embedding to discern
	#	information from the position within the text. This is
	#	important as there is no recurrence in a transformer model, so
	#	there is no inherent understanding of the word's position.
	# -> a sentence embedding is added to each token (further
	#	differentiating between sentences).
	# To perform the classification for Next Sentence Prediction, the
	# output of the [CLS] embedding, which denotes the "aggregate
	# sequence representation for sentence classification" is passed
	# through a classification layer with softmax to return the
	# probability of the two sentences being sequential.
	# Because this current task is classification, to fine-tune BERT,
	# one needs to simply add classification layers over BERT's outputs
	# of the [CLS] token.


	# Implementing BERT
	# Use Tensorflow hub's BERT preprocesser and encoder. Do not pass
	# the text through the frameword described earlier (which removes
	# capitalization, applies lemmatization, etc). This has been
	# abstracted with the BERT preprocessor.
	data = total[["title", "class"]]

	# Split the model into training and testing data with an 80:20
	# train:test ration.
	titles = data["title"].values
	labels = data["class"].values
	title_train, title_test, y_train, y_test = train_test_split(
		titles, labels, test_size=0.2, random_state=1000
	)

	bert_preprocess = hub.KerasLayer(
		"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
	)
	bert_encoder = hub.KerasLayer(
		"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
	)

	# Now we can develop the neural network. It must be a functional
	# model, where the output of a previous layer must be an argument
	# to the next layer. The model consists of:
	# -> 1 Input layer: This will represent the sentence that will be
	#	passed into the model.
	# -> The bert_preprocess layer: Here, we pass in the input to
	#	preprocess the text.
	# -> The bert_encoder layer: Here, we pass the preprocessed tokens
	#	into the BERT encoder.
	# -> 1 Dropout layer with a rate of 0.2. The pooled_output of the
	#	BERT encoder is passed into it (more on this below).
	# -> 2 Dense layers with 10 and 1 neurons respectively. The first
	#	one will use a ReLU activation function, and the second will
	#	use sigmoid.
	input_layer = tf.keras.layers.Input(
		shape=(), dtype=tf.string, name="news"
	)

	# BERT layers.
	processed = bert_preprocess(input_layer)
	output = bert_encoder(processed)

	# Fully connected layers.
	layer = tf.keras.layers.Dropout(
		0.2, name="dropout"
	)(output["pooled_output"])
	layer = tf.keras.layers.Dense(
		10, activation="relu", name="hidden"
	)(layer)
	layer = tf.keras.layers.Dense(
		1, activation="sigmoid", name="output"
	)(layer)

	# Model.
	model = tf.keras.Model(inputs=[input_layer], outputs=[layer])

	# The "pooled_output" will be passed into the dropout layer. This
	# value denotes the overall sequence representation of the text. As
	# mentioned earlier, it is the representation of the [CLS] token
	# outputs.

	# Use the Adam optimizer, binary crossentropy loss, and a
	# performance metric of accuracy to train the model. The model will
	# be trained over 5 epochs. The hyperparameters can be adjusted.
	model.compile(
		optimizer="adam",
		loss="binary_crossentropy",
		metrics=["accuracy"],
	)

	# This model will have a maximum accuracy of around 91.1% on the
	# test data.
	model.fit(title_train, y_train, epochs=5)
	model.evaluate(title_test, y_test)


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()