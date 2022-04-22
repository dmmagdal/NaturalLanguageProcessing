# extractive_summarization.py
# Use techniques such as jaro winkler distance and page rank to perform
# extractive text summarization.
# Source: https://towardsdatascience.com/text-summarization-in-python-
# with-jaro-winkler-and-pagerank-72d693da94e8
# Source (Jaro-Winkler Wikipedia): https://en.wikipedia.org/wiki/
# Jaro-Winkler_distance
# # Source (C wikipedia): https://en.wikipedia.org/wiki/Python_
# (programming_language)
# Source (Python wikipedia): https://en.wikipedia.org/wiki/C_
# (programming_language)
# Windows/MacOS/Linux
# Python 3.7


import string
import jaro
import networkx as nx
import numpy as np
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")


def main():
	# Introduction to Text Summarization
	# There are two main approaches of text summarization, extractive
	# and abstractive. Extractive solutions require selecting specific
	# sentences from the body of text to generate the final summary.
	# The general approach to extractive solutions are associated with
	# ranking their importance in the body of the text and returning
	# the most important sentences. Abstractive solutions to text
	# summarization involves creating new sentences to capture context
	# and meaning behind the original body of the text. Text
	# compression techniques are commonly used to solve abstractive
	# approaches in text summarization.
	# This example will build an algorithmic text summarizer using the
	# extractive approach, relying on two main algorithms: the Jaro-
	# Winkler distance to measure the distance between a pari of
	# sentences and the page rank algorithm which will rank the
	# sentences based on their influence in the network.


	# Difficulties of Text Summarization
	# There are a variety of factors in text summarization which
	# changes the impact the summary might have on the original story.
	# Here are two components that make summarization a very difficult
	# task in NLP:
	# -> What is the ideal number of sentences that the summary must 
	#	hold? A number too large night be useless since you're 
	#	practically reading the entire body of text. A number too small
	#	might have large consequences in the summary (ie skipping 
	#	important details and plot lines). Shorter summaries hold less
	#	information.
	# -> How can you give context in the summary? When providing
	#	summaries of subplots within a story, context is the most
	#	important part. It's what makes the summary useful. This problem
	#	is more prominent in extractive solutions than abstractive.


	# Architecture Overview
	# This extractive solution will have 6 steps:
	# 1) Get the input text.
	# 2) Clean the input text of certain punctuation, stopwords, etc.
	# 3) Generate a sentence similarity adjacency matrix.
	# 4) Create the sentence network (nodes are sentences and edges
	#	hold the similarity of a pair of sentences).
	# 5) Rank the nodes in the network using page rank or other ranking
	#	measures.
	# 6) Generate the summary based on the top N ranked nodes 
	#	(sentences).


	# Problem Statement
	# Given a body of text, create a pipeline which will generate a
	# summary of the input body of the text.


	# Load Text
	python_wiki_text = """
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small- and large-scale projects.
Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.
Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000 and introduced new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support. Python 3.0, released in 2008, was a major revision that is not completely backward-compatible with earlier versions. Python 2 was discontinued with version 2.7.18 in 2020.
Python consistently ranks as one of the most popular programming languages.
"""

	c_wiki_text = """
C (/ˈsiː/, as in the letter c) is a general-purpose computer programming language. It was created in the 1970s by Dennis Ritchie and Bell Labs, and remains very widely used and influential. By design, C's features cleanly reflect the capabilities of the targetted CPUs. It has found lasting use in operating systems, device drivers, protocol stacks, though decreasingly for application software, and is common in computer architectures that range from the largest supercomputers to the smallest microcontrollers and embedded systems.
A successor to the programming language B, C was originally developed at Bell Labs by Dennis Ritchie between 1972 and 1973 to construct utilities running on Unix. It was applied to re-implementing the kernel of the Unix operating system. During the 1980s, C gradually gained popularity. It has become one of the most widely used programming languages, with C compilers available for almost all modern computer architectures and operating systems. C has been standardized by ANSI since 1989 (ANSI C) and by the International Organization for Standardization (ISO).
C is an imperative procedural language supporting structured programming, lexical variable scope, and recursion, with a static type system. It was designed to be compiled to provide low-level access to memory and language constructs that map efficiently to machine instructions, all with minimal runtime support. Despite its low-level capabilities, the language was designed to encourage cross-platform programming. A standards-compliant C program written with portability in mind can be compiled for a wide variety of computer platforms and operating systems with few changes to its source code.
Since 2000, C has consistently ranked among the top two languages in the TIOBE index, a measure of the popularity of programming languages.
"""

	# Constants.
	sw = list(set(stopwords.words("english")))
	punct = list(string.punctuation.replace(".", ""))


	# Clean Text
	# For this, we are primarily going to remove stopwords and certain
	# punctuations so the procedures later on in the pipeline can be
	# more efficient and accurate in their calculation.
	def clean_text(text, sw=sw, punct=punct):
		article = text.lower()

		for pun in punct:
			article = article.replace(pun, "")

		article = article.replace("[a-zA-Z]", " ")\
			.replace("\r\n", " ")\
			.replace("\n", " ")
		original_text_mapping = {
			k:v for k,v in enumerate(article.split(". "))
		}

		article = article.split(" ")

		article = [
			x.lstrip().rstrip() for x in article if x not in sw
		]
		article = [x for x in article if x]
		article = " ".join(article)

		return original_text_mapping, article


	original_text_mapping1, cleaned_text1 = clean_text(
		python_wiki_text
	)
	original_text_mapping2, cleaned_text2 = clean_text(c_wiki_text)

	sentences1 = [
		x for x in cleaned_text1.split(". ")
		if x not in ["", " ", "..", ".", "..."]
	]
	sentences2 = [
		x for x in cleaned_text2.split(". ")
		if x not in ["", " ", "..", ".", "..."]
	]
	print("Number of sentences in text1:", len(sentences1))
	print("Number of sentences in text2:", len(sentences2))


	# Create Network
	# This section creates an adjacency matrix through similarity of
	# various sentences. The similarity of sentences will be
	# calculated using the Jaro-Winkler distance. The output of this
	# distance measures will be a floating point number between the
	# values of 0 and 1. The closer the number is to 1 indicates the
	# more similar the pair of sentences are. The closer the number is
	# to 0 indicates the more the pair of sentences are not similar.
	# The adjacency matrix will then allow us to create a weighted
	# network (through networkx). Keep in mind the size of the input
	# body of text since the similarity matrix needs to assign a score
	# for each pair of sentences it will be an extensive and exhaustive
	# process for the computer if the body of text is very large
	# (> 2000 sentences).
	# In the created network, the nodes will be indices associated to
	# the sentences in the book and the edges connecting the sentences
	# will be weighted based on the similarity of a pair of sentences.
	# We can then run the page rank algorithm on that weighted network
	# to identify nodes with a large rank associated to them.
	def create_similarity_matrix(sentences):
		sentence_length = len(sentences)
		sim_mat = np.zeros((sentence_length, sentence_length))

		for i in range(sentence_length):
			for j in range(sentence_length):
				if i != j:
					similarity = jaro.jaro_winkler_metric(
						sentences[i], sentences[j]
					)
					sim_mat[i][j] = similarity
		return sim_mat


	# The built in page rank function will output a dictionary where
	# the keys are the nodes (sentence index in this case) and the
	# values are the associated page rank score to that node. Keep in
	# mind tahta we want to map the sentence index back to the original
	# sentence and not the cleaned one used to create the network. This
	# way, the resulting summary generated will be much easier to
	# interpret when including the stop words, punctuations, etc.
	sim_mat1 = create_similarity_matrix(sentences1)
	G1 = nx.from_numpy_matrix(sim_mat1) # create network
	pr_sentence_similarity1 = nx.pagerank(G1) # calculate page rank scores
	ranked_sentences1 = [
		(original_text_mapping1[sent], rank)
		for sent, rank in sorted(
			pr_sentence_similarity1.items(),
			key=lambda item: item[1],
			reverse=True
		)
	]

	sim_mat2 = create_similarity_matrix(sentences2)
	G2 = nx.from_numpy_matrix(sim_mat2)
	pr_sentence_similarity2 = nx.pagerank(G2)
	ranked_sentences2 = [
		(original_text_mapping2[sent], rank)
		for sent, rank in sorted(
			pr_sentence_similarity2.items(),
			key=lambda item: item[1],
			reverse=True
		)
	]

	print(ranked_sentences1[0][0])
	print(ranked_sentences2[0][0])


	# Generate Summary
	# Now that we have the ranked sentences based on their page rank
	# score, we can generate a summary given the users input of the
	# sentences they want in the summary.
	def generate_summary(ranked_sentences, N):
		summary = ". ".join([sent[0] for sent in ranked_sentences[0:N]])
		return summary


	N = 3 #25
	summary1 = generate_summary(ranked_sentences1, N)
	summary2 = generate_summary(ranked_sentences2, N)

	print("-"*72)
	print("Python wiki article:")
	print(python_wiki_text)
	print("Python summary:")
	print(summary1)
	print("="*72)
	print("C wiki article:")
	print(c_wiki_text)
	print("C summary:")
	print(summary2)
	print("-"*72)


	# Conclusion
	# The result of the summary would vary from text to text, certain
	# bodies of text might be better than other for the given task.


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()