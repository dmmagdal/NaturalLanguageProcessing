# tf_pegasus_summarizer.py
# Perform (abstractive) text summarization with the Pegasus model using
# Huggingface transformers. The major difference is that this uses the
# Tensorflow version of the model.
# Source: https://www.youtube.com/watch?v=Yo5Hw8aV3vY
# Source (Huggingface documentation): https://huggingface.co/docs/
# transformers/model_doc/pegasus
# Source (C wikipedia): https://en.wikipedia.org/wiki/Python_
# (programming_language)
# Source (Python wikipedia): https://en.wikipedia.org/wiki/C_
# (programming_language)
# Windows/MacOS/Linux
# Python 3.7


from transformers import TFPegasusForConditionalGeneration, PegasusTokenizer


def main():
	# Huggingface pegasus models (google only):
	# google/pegasus-xsum 						https://huggingface.co/google/pegasus-xsum
	# google/pegasus-large 						https://huggingface.co/google/pegasus-large
	# google/pegasus-cnn_dailymail 				https://huggingface.co/google/pegasus-cnn_dailymail
	# google/pegasus-multi_news 				https://huggingface.co/google/pegasus-multi_news
	# google/bigbird-pegasus-large-arxiv 		https://huggingface.co/google/bigbird-pegasus-large-arxiv
	# google/bigbird-pegasus-large-bigpatent	https://huggingface.co/google/bigbird-pegasus-large-bigpatent
	# google/bigbird-pegasus-large-pubmed 		https://huggingface.co/google/bigbird-pegasus-large-pubmed
	# google/pegasus-newsroom 					https://huggingface.co/google/pegasus-newsroom
	# google/pegasus-pubmed 					https://huggingface.co/google/pegasus-pubmed
	# google/pegasus-arxiv 						https://huggingface.co/google/pegasus-arxiv
	# google/pegasus-big_patent 				https://huggingface.co/google/pegasus-big_patent
	# google/pegasus-wikihow 					https://huggingface.co/google/pegasus-wikihow
	# google/pegasus-billsum 					https://huggingface.co/google/pegasus-billsum
	model = "google/pegasus-xsum"
	cache_dir = "./pegasus-xsum"

	# Download/load pre-trained model and tokenizer.
	tokenizer = PegasusTokenizer.from_pretrained(model, cache_dir=cache_dir)
	pegasus = TFPegasusForConditionalGeneration.from_pretrained(model, cache_dir=cache_dir)

	# Input text to summarize. Here we're just using the Python and C
	# programming language wikipedia texts (in particular, the first
	# few paragraphs for each).
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

	# Tokenize the input texts.
	python_tokens = tokenizer(
		python_wiki_text, 
		truncation=True, # shorten text to be the appropriate length for model.
		padding="longest", # want the padding to be as long as possible
		return_tensors="tf", # return tensorflow tensors.
	)
	c_tokens = tokenizer(
		c_wiki_text,
		truncation=True,
		padding="longest",
		return_tensors="tf",
	)

	# Summarize texts.
	python_summary = pegasus.generate(**python_tokens)
	c_summary = pegasus.generate(**c_tokens)
	
	# Decode each summary.
	print("Python wiki article:")
	print(python_wiki_text)
	print("Python summary:")
	print(tokenizer.decode(python_summary[0])) # Grab first instance of result.
	print("="*72)
	print("C wiki article:")
	print(c_wiki_text)
	print("C summary:")
	print(tokenizer.decode(c_summary[0]))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()