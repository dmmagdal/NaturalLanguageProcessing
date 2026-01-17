# rake_scratch.py
# Implement RAKE from scratch.
# Source: https://www.youtube.com/watch?v=ZOgrhn2Uq0U&ab_channel=1littlecoder
# Windows/MacOS/Linux
# Python 3.11


from datasets import load_dataset
import nltk
import numpy as np
import pandas as pd
from rake_nltk import Rake


def main():
	required_ntlk_downloads = [
		"stopwords", "punkt_tab"
	]
	for download in required_ntlk_downloads:
		nltk.download(download)

	# Dataset link: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes
	# Alternative link: Sadaftb/clinical-nlp-patient-notes
	data = load_dataset("Sadaftb/clinical-nlp-patient-notes", split="train")
	notes = data.to_pandas()
	print(notes.pn_history[10])
	print()

	# General steps to Rake:
	# 1. Split (tokenize) text and remove stop words.
	# 
	# 2. We will use the places whether there is a stop word as a sort 
	# of "word boundary". This is how our candidate phrases are 
	# created.
	# 
	# 3. Next, count all the (remaining) words.
	# 
	# 4. For rake, you need the following metrics:
	# - word frequency (as seen above)
	# - degree of word (how many times a word co-occurs)
	# - degree score (degree of word / word frequency)
	# 
	# 5. For each degree score, we're taking the sum of the co 
	# occurences across all words.
	# 
	# 6. Sum the degree score for each word in a word phrase and sort 
	# (largest to smallest).
	# 
	# 7. The output of rake should be the word phrase and associated 
	# scores.
	# NOTE:
	# Plenty of people use rake for a topic modeling context/task.

	# Initialize rake.
	r = Rake(
		punctuations=[
			')', '(', '.', ',', ':', '}', '{', ').', '}.', '),', '},'
		] # Define what punctuations to look for/clean/remove.
	)

	# Perform keyword extraction.
	r.extract_keywords_from_text(notes.pn_history[20])

	# Print the ranked phrases.
	print(r.get_ranked_phrases())
	print()

	# Get ranked phrases with scores (in a dataframe).
	phrases_df = pd.DataFrame(
		r.get_ranked_phrases_with_scores(), columns=["score", "phrase"]
	)
	print(phrases_df.loc[phrases_df.score > 3])

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()