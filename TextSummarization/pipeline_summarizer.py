# pipeline_summarizer.py
# Perform (abstractive) text summarization with the Pegasus model using
# Huggingface transformers.
# Source: https://www.youtube.com/watch?v=TsfLm5iiYb4
# Source (Huggingface documentation): https://huggingface.co/docs/
# transformers/v4.18.0/en/main_classes/pipelines#transformers.
# SummarizationPipeline
# Source (C wikipedia): https://en.wikipedia.org/wiki/Python_
# (programming_language)
# Source (Python wikipedia): https://en.wikipedia.org/wiki/C_
# (programming_language)
# Windows/MacOS/Linux
# Python 3.7


from transformers import pipeline


def main():
	# Initialize a generic summarization pipeline. You can set up the
	# pipeline to use specific models and tokenizers (refer to the 
	# linked huggingface documentation above). Default model used is
	# sshleifer/distilbart-cnn-12-6.
	summarizer = pipeline("summarization")

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

	# Summarize texts.
	python_summary = summarizer(
		python_wiki_text, # input text
		max_length=256, # maximum number of tokens/length of the summary
		min_length=32, # maximum number of tokens/length of the summary
		do_sample=False, # use greedy decoder if False
	)
	c_summary = summarizer(
		c_wiki_text,
		max_length=256,
		min_length=32,
		do_sample=False,
	)

	# Print the summaries.
	print("Python wiki article:")
	print(python_wiki_text)
	print("Python summary:")
	for i in python_summary:
		print(i["summary_text"])
		print("-"*72)
	print("="*72)
	print("C wiki article:")
	print(c_wiki_text)
	print("C summary:")
	for i in c_summary:
		print(i["summary_text"])
		print("-"*72)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()