'''
A "simplified search engine. will load a set of documents (the "reuters" corpus)
and given a queries list, will provide a relevant documents

Created on Nov 24, 2014

@author: nancy and michal
'''
import csv
import numpy as np
import os
import nltk
from nltk.corpus  import reuters, stopwords
from nltk import word_tokenize
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer

# input/output files paths
global queriesFilePath; queriesFilePath = '/home/RecSys/queries.txt'
global resultsFilePath; resultsFilePath = '/home/RecSys/results.txt'

class Index:
    """ Inverted index data structure """
 
    def __init__(self):
        """
        class initialization:
        tokenizer- NLTK compatible tokenizer function
        stemmer- NLTK compatible stemmer 
        stop_words- list of ignored words
        lemm- NLTK compatible lemmatizer
        inv_index- (defaultdict) the inverted index
        positional_index- (defaultdict of defaultdicts) relevant for the bonus task only
        """
        # Tokenization
        self.tokenizer = word_tokenize
        
        # Stemming
        self.stemmer = EnglishStemmer()
        #self.stemmer = nltk.PorterStemmer()
        #self.stemmer = nltk.LancasterStemmer()
        
        # Stopwords
        self.stop_words = stopwords.words('english')
        
        # Lemmatization
        self.lemm = nltk.WordNetLemmatizer()
        
        # The invereted index
        self.inv_index = defaultdict(list)
        # The positional index (for the bonus task)
        self.positional_index = defaultdict(lambda: defaultdict(list))
        
    def loadReutersCorpus(self):
        '''  
        Load the Reuters datasets.
        Note: need to install nltk data beforehand, using nltk.download() for the first run
        '''       
        self.fileIds = reuters.fileids()
        # Use only the training set corpus - documents located in the reuters/training folder.
        self.fileIds = [x for x in self.fileIds if x.startswith( 'training' )]
        return
        
    def analyzeDoc(self,rawText):      
        ''' 
        Processing raw data: tokenization, stopwords removal, stemming and lemmatization
        
        Parameters
        ----------
        rawText : unicode
            The document full text.
        
        Returns
        -------
        filtered_tokens : [w1, w2, ...,wn] list
            
        '''
        # Tokenization
        tokens = self.tokenizer(rawText)
       
        # Stop-words removal and convert to lowercase
        filtered_tokens = [w.lower() for w in tokens if not w.lower() in self.stop_words]
        
        # Stemming
        filtered_tokens = [self.stemmer.stem(t) for t in filtered_tokens]
        
        # Lemmatization
        filtered_tokens = [self.lemm.lemmatize(t) for t in filtered_tokens]
            
        return filtered_tokens

    def addDocToInvertedIndex(self, docId):
        """
        Add document tokens to the index
        
        Parameters
        ----------
        docId : string
            The document id.
        
        Returns
        -------
        None
        """
        # Retrieve the document content by its id
        rawDoc = self.getDocContentById(docId)
        
        # Analyze the document - i.e., tokenization, stemming, and stop-words removal.
        tokens = self.analyzeDoc(rawDoc)
        
        # remove prefix
        docId = docId[len('training/'):]
        
        # Add the tokens to the Inverted Index
        for token in tokens:
            if docId not in self.inv_index[token]:
                self.inv_index[token].append(docId)
                
        return 
    
    def addDocToPositionalIndex(self, docId):
        """
        Add a document, and all term appearance (positions), to the positional index
        
        Parameters
        ----------
        docId : string
            The document id.
        
        Returns
        -------
        None
        """
        # Retrieve the document content by its id
        rawDoc = self.getDocContentById(docId)
        
        # Analyze the document - i.e., tokenization, stemming, and stop-words removal.
        tokens = self.analyzeDoc(rawDoc)
        
        # Remove prefix
        docId = docId[len('training/'):]
        
        # Add the tokens to the positional Index
        for token in tokens:
            # If we havn't processed this term in this doc already
            if docId not in self.positional_index[token]:
                # Find all occurrences (positions) of this term in the document
                indices = np.where(np.asarray(tokens) == token)[0]
                # Update the index with the positions list
                self.positional_index[token][docId].append(indices)
                
        return
    
    def getDocContentById(self, docId):
        ''' 
        Gets the document content by its id
        
        Parameters
        ----------
        docId : string
            The document id.
        
        Returns
        -------
        The document content (unicode)
        '''
        return reuters.raw(docId)
        
    def buildInvertedIndex(self):
        '''Build the inverted index'''
        # Run over all files in corpus and handle each one of them separately
        for fileId in self.fileIds:
            self.addDocToInvertedIndex(fileId)
        return
    
    
    def buildPositionalIndex(self):
        '''Build the positional index'''
        # Run over all files in corpus and handle each one of them separately
        for fileId in self.fileIds:
            self.addDocToPositionalIndex(fileId)
            
        # The below command exports the positional index member- FOR TESTING ONLY 
        # DEBUG: writeDictToFile('/home/RecSys/positional_results.txt',self.positional_index)
          
class Queries:
    
    def __init__(self, filePath, inv_index):
        '''
        Class initialization
        
        Parameters
        ----------
        filePath : string
            The queries input file path
        inv_index: defaultdict
            The corpus inverted index
        
        Returns
        -------
        None        
        '''
        self.queries_list = []  
        self.loadQueriesFile(filePath)
        self.inv_index = inv_index

    
    def loadQueriesFile(self, filePath):
        """
        Read queries from input file
        
        Parameters
        ----------
        filePath : string
            The queries file full path.
        
        Returns
        -------
        None
        
        """
        with open(filePath,"r") as queries:      
            for query in queries:
                query = query.rstrip()
                self.queries_list.append([str(s) for s in query.split(';')])
        return
    
    def queryProcessing(self,filePath):
        """
        Process the boolean AND query of all terms. Finds all relevant documents and writes the answer to file
        
        Parameters
        ----------
        filePath : string
            The output file full path
        
        Returns
        -------
        None
        """
        for query in self.queries_list:
            
            relevent_docs = []      # intersection of documents (AND)
            docs = []               # docs that contain query terms
            
            # Merge terms to one string
            text = ' '.join(query)
            
            # Analyze query (same actions as document analysis: tokenizer, stemmer, etc)
            filtered_query = Index().analyzeDoc(text)
            
            # Find query terms in inverted index
            for term in filtered_query:
                # lookup documents that this term appears at
                docs.append(set(self.inv_index[term])) # we used casting to 'set' for using 'intersection' later on
                
            # AND conditioning
            if len(docs):   # if docs not empty
                relevent_docs = set(docs[0]).intersection(*docs[1:])
            
            # Export the list of relevant documents to the output file
            # DEBUG: print "The relevant doc ids, for query: " +  str(text) + " are: " + str(sorted(relevent_docs))
            writeToFile(filePath, relevent_docs)
            
        return

def writeToFile(filePath, text):
    '''
    Export a list separated by ';' to output file.
    If the list is empty exports 'no results'
    
    Parameters
    ----------
    filePath : string
        The destination file full path.
    text : list (possible unicode elements)
    
    Returns
    -------
    None
    '''
    f = open(filePath, "a")
    w = csv.writer(f)
    if len(text):   # if text not empty
        w.writerow([';'.join(sorted(list(text)))])    # we used 'list' casting cause text is an array of unicode elemants
    else:           # text is empty
        w.writerow(['no results'])
    f.close()
    return

def writeDictToFile(filePath,d):
    '''
    Export a dictionary to output file
    
    Parameters
    ----------
    filePath : string
        The destination file full path.
    d : dictionary
    
    Returns
    -------
    None
    '''
    f = open(filePath, "a")
    w = csv.writer(f)
    
    for key, val in d.items():
        w.writerow([key, val])
        
    f.close()
    return

def main():
    
    # Input/output file paths
    global queriesFilePath,resultsFilePath
    
    # Delete the results file at each run- for multiple executions
    os.path.exists(resultsFilePath) and os.remove(resultsFilePath) 
    
    index = Index() # Index instantiation
    
    # Load documents from the given dataset.
    index.loadReutersCorpus()
    
    # Build the inverted index
    index.buildInvertedIndex()
    
    # Process boolean query and retrieve the relevant documents 
    queries = Queries(queriesFilePath, index.inv_index)
    queries.queryProcessing(resultsFilePath)
    
    # Build the positional index - bonus task:
    index.buildPositionalIndex()

    return
    

main()    
    