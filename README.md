# Natural Language Processing

Description: This (mono)repo is meant to encapsulate all NLP projects that may have future applications on other projects.


### Repo components

 - **DetectFakeNews**: An example using BERT and LSTM models to determine which news articles are fake news (by title).
 - **NeuralSearch**: A submodule that focuses on indexing and searching texts with techniques such as TF-IDF, text similarity with levenshtein diestance, and BERT embeddings.
 - **TextSummarization**: An example using primarily Pegasus model to summarize texts.


### Updating the Repo

Run the following commands to pull updates from GitHub:
 - `git pull`
 - `git submodule update --init --recursive`

To push data from the main repo, simply add, commit, and update like normal:
 - `git add files`
 - `git commit -m "commit message"`
 - `git push`

To update data from a submodule, make all changes from the submodule repository. The proceed to update the main repo by calling the submodule update command.


#### References

- [Creating git submodules](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/How-to-add-submodules-to-GitHub-repos)