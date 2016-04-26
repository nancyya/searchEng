Notes on the execution of the code:

This code had been implemneted on Unix OS. python 2.7 (anaconda platform).

Define the directories of the input/output files by the following global variables:
    queriesFilePath;
    resultsFilePath;
 
Install the nltk package. 
Run the nltk.download() command (for first run only).
Then: 
- Download corpuses:
    1. reuters
    2. stopwords 
    3. wordnet
- Download Models: 
    1. punkt (tokenizer) 
    2. rslp (stemmer) 
    
The main() functions runs the relevant procedures:
1. Deleting the "resultsFilePath" file if exists (in case that it is not the first run)
2. loadReutersCorpus() - auto loads documents from the given dataset (installing nltk data!).
3. buildInvertedIndex() - builds the inverted index
4. Queries() - process boolean query and retrieve the relevant documents.
    loadQueriesFile() - loads queries from the given queries.txt file 
    queryProcessing() - calculates the relevent files and writes them (sorted by id) into results.txt file, as requested.
    Note: if all the query terms do not appear in any document 'no results' is returned.
5. (bonus task) buildPositionalIndex() - builds the positional index. 
    Please note that the query processing part was not implemented.


Additional notes:
1. Stemming - in Index.__init__() you can change the stemmer type: Snowball/Porter/Lancaster