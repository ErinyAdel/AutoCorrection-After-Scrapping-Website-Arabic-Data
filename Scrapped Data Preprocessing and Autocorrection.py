# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:51:18 2021
@author: Eriny
"""

import requests
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from collections import Counter
import numpy as np
import pandas as pd
import csv

header = ['اسم الكاتب', 'عنوان الكتاب', 'نوع الكتاب', 'رابط الكتاب']
extracted_data_file     = './extracted_data.csv'
cleaned_data_file       = './cleaned_data.csv'
autocorrected_data_file = './autocorrected_extracted_data.csv'
dictionary_file         = './arabic-wordlist-1.6.txt'
delimiter = '،،،،'

# ---------------------------------------------------------------------------- #

### Step 1) Scrapping
class ScrapData(object):
    
    def __init__(self, file_path):
        print("Start Scrapping...") 
        ## 
        page_num       = 1
        authorsList    = []
        titlesList     = []
        categoriesList = []
        linksList      = []
        
        while page_num <= 200:
            ## Use Request To Fetch The URL
            result = requests.get(f"https://www.arab-books.com//page/{page_num}")
            
            ## Save Page Content/Markup
            src = result.content
            
            ## Create Soup Object To Parse Content
            soup = BeautifulSoup(src, "lxml")
            
            ## Find The Elements Containing --> Author, Title, Link To Pdf, 
            titles  = soup .find_all("div", {"class":"excerpt-book"})

            ## Loop Over Returned Lists To Extract Needed Content
            for i in range(len(titles)):
                if titles[i].text == '\n':
                    authorsList.append('غير معروف')
                else:
                    titlesList.append(titles[i].text.replace("\n", ""))                    

                linksList.append(titles[i].find("a").attrs['href'])
          
            print("Page", page_num, "Switched")    
            page_num += 1
        
            
        ## Extract Info From Each Link
        for i, link in enumerate(linksList):
            print(f"Extract Data From Book {i}....")
            try:
                result = requests.get(link)
                src = result.content
                soup = BeautifulSoup(src, "lxml")
                authors = soup.find("div", {"class":"book-info"})      
                authors.find("ul")
                for li in authors.find_all("li"):
                    if li.find("a"):
                        authors = re.sub('\n*', '', li.find("a").text)
                        break
                    else:
                        authors = 'غير معروف'  
                        break
            except:
                authors = 'غير معروف'  
            
            try:
                result = requests.get(link)
                src = result.content
                soup = BeautifulSoup(src, "lxml")
                details = soup.find("div", {"class":"book-info"})
                i = 0
                details.find("ul")
                for li in details.find_all("li"):
                    li.find_all("a")
                    if i == 1:
                        if li.find("a"):
                            details = re.sub('\n*', '', li.find("a").text)
                            break
                        else:
                            details = 'غير معروف'  
                            break
                    i += 1
            except:
                details = 'غير معروف'  

            authorsList.append(authors)
            categoriesList.append(details)     

        ## Create txt File & Fill it With Extracted Values            
        print("-- SAVING Extracted Data In csv File --")
        with open(file_path, "w", encoding='utf-8-sig') as file:
            wr = csv.DictWriter(file, header)
            wr.writeheader()
            
            for i in range(len(linksList)):
                s = u','.join([str(authorsList[i]), str(titlesList[i]), 
                               str(categoriesList[i]), str(linksList[i])]) + u'\n'
                file.write(s)
                             
        print("Scrapping Finished!\n")    

# ---------------------------------------------------------------------------- #

### Step 2) Cleaning

class CleanData(object):
    
    def __init__(self, authors, titles, categories, links):
        print("Start Cleaning....") 
        filtered_authors    = self._clean(authors)
        filtered_titles     = self._clean(titles)
        filtered_categories = self._clean(categories)
        ## Write Cleaned Scrapped Data Into New Text File
        print("-- SAVING Cleaned Data In csv File --")
        with open(cleaned_data_file, "w", encoding='utf-8-sig') as file:
            wr = csv.writer(file)
            wr.writerow(header)
            lines1 = filtered_authors.split(delimiter)
            lines2 = filtered_titles.split(delimiter)
            lines3 = filtered_categories.split(delimiter)
            
            for (l1, l2, l3, l4) in zip(lines1, lines2, lines3, links):
                s = u','.join([str(l1), str(l2), str(l3), str(l4)]) + u'\n'
                file.write(s)
                
        print("Cleaning Finished!\n")  
      

    def _clean(self, text):
        ## 1.1 Replace punctuations with a white space
        remove_punctuations = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        ## 1.2 Normalize
        normalized = re.sub("گ", "ك", remove_punctuations)
        normalized = re.sub("ى", "ي", normalized)
        ## 2.1 Remove Non-Arabic Words
        remove_nonarabic = re.sub(r'\s*[A-Za-z]+\b', ' ' , normalized)
        ## 2.2 Remove Stop Words.
        stop_words = set(stopwords.words('arabic'))
        filtered_sentence = ' '.join([word for word in remove_nonarabic.split() 
                                          if word not in stop_words])
        ## 3.0 Stemming
        st = ISRIStemmer()
        st.stem(filtered_sentence)
        
        return filtered_sentence
    
# ---------------------------------------------------------------------------- #

class SpellChecker(object):

    def read_corpus(self, filename):
        with open(filename, "r", encoding='utf-8-sig') as file:
            lines = file.readlines() ## Read The File Line By Line
            words = []
            for line in lines:
              words += re.findall(r'\w+', line.lower()) ## Put Each Word In Its Lowercase.
    
        return words
    
    def read_corpus_lines(self, filename):
        lines1 = []
        lines2 = []
        lines3 = []
        lines4 = []
        with open(filename, "r", encoding='utf-8-sig') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                lines1.append(row["اسم الكاتب"])
                lines2.append(row["عنوان الكتاب"])
                lines3.append(row["نوع الكتاب"])
                lines4.append(row["رابط الكتاب"])
        return lines1, lines2, lines3, lines4
    
    def get_count(self, word_list):
        word_count_dict = {}  ## Each Word Count
        word_count_dict = Counter(word_list)
        return word_count_dict
    
    def get_probs(self, word_count_dict):
        ## 𝑃(𝑤ᵢ) = 𝐶(𝑤ᵢ) / M 
        m = sum(word_count_dict.values())
        word_probs = {w: word_count_dict[w] / m for w in word_count_dict.keys()}
        
        return word_probs
    
    def _split(self, word):
        return [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    def _delete(self, word):
        return [l + r[1:] for l,r in self._split(word) if r]
    
    def _swap(self, word):
        return [l + r[1] + r[0] + r[2:] for l, r in self._split(word) if len(r)>1]
    
    def _replace(self, word):
        letters = string.ascii_lowercase
        return [l + c + r[1:] for l, r in self._split(word) if r for c in letters]
    
    def _insert(self, word):
        letters = string.ascii_lowercase
        return [l + c + r for l, r in self._split(word) for c in letters]
       
    def _edit1(self, word):  
        return set(self._delete(word) + self._swap(word) + 
                   self._replace(word) + self._insert(word))
    
    def _edit2(self, word):
      return set(e2 for e1 in self._edit1(word) for e2 in self._edit1(e1))
    
    def correct_spelling(self, word, vocabulary, word_probability):
        if word in vocabulary:
            #print(f"\n'{word}' is already correctly spelt")
            return 
        
        suggestions = self._edit1(word) or self._edit2(word) or [word]
        best_guesses = [w for w in suggestions if w in vocabulary]
          
        return [(w, word_probability[w]) for w in best_guesses]
            
    
    def correct_word(self, word, corrections):
        if corrections:
            print('\nSuggested Words:', corrections)
            probs = np.array([c[1] for c in corrections])
            ## Get The Index Of The Best Suggested Word (Higher Probability)
            best_ix = np.argmax(probs) 
            correct = corrections[best_ix][0]
            print(f"\n'{correct}' is Suggested for '{word}'")
            return correct        
        
# ---------------------------------------------------------------------------- #

### Scrapping 
ScrapData('extracted_data.csv')

### Cleaning
def split_file(filename):
    data = pd.read_csv(filename)
    
    # converting column data to list
    authors = data['اسم الكاتب'].tolist()
    titles = data['عنوان الكتاب'].tolist()
    categories = data['نوع الكتاب'].tolist()
    links = data['رابط الكتاب'].tolist()

    return authors, titles, categories, links

authors, titles, categories, links = split_file(extracted_data_file)

authors = list(map(lambda s: s+delimiter, authors))
titles = list(map(lambda s: s+delimiter, titles))
categories = list(map(lambda s: s+delimiter, categories))

CleanData(str(authors), str(titles), str(categories), links)        

### AutoCorrection
def autocorrect_misspellings(lines, vocabs, word_prob):
    correct_lines = []
    for line in lines:
        for word in line.split(' '):
            corrections = spell_checker.correct_spelling(word, vocabs, word_prob)
            correct     = spell_checker.correct_word(word, corrections)
            if correct:
                correct_lines.append(correct)
            else:
                correct_lines.append(word)

    correct_lines = ' '.join(correct_lines)
    
    return correct_lines

spell_checker = SpellChecker()
print("Starting AutoCorrecting Misspelling....\n")
base_words = spell_checker.read_corpus(dictionary_file)
vocabs = set(base_words) ## Vocabulary (Unique Words)

word_dict_counts = spell_checker.get_count(base_words)
word_prob   = spell_checker.get_probs(word_dict_counts) 

authors, titles, categories, links = spell_checker.read_corpus_lines(cleaned_data_file)   
authors = list(map(lambda orig_string: orig_string+'،،،،', authors))   
titles = list(map(lambda orig_string: orig_string+'،،،،', titles))   
categories = list(map(lambda orig_string: orig_string+'،،،،', categories))   

correct_authors    = autocorrect_misspellings(authors, vocabs, word_prob)
correct_titles     = autocorrect_misspellings(titles, vocabs, word_prob)
correct_categories = autocorrect_misspellings(categories, vocabs, word_prob)

with open(autocorrected_data_file, "w", encoding='utf-8-sig') as file:
    print("-- SAVING Autocorrected Data In csv File --")
    wr = csv.writer(file)
    wr.writerow(header)
    
    lines1 = correct_authors.split(delimiter)
    lines2 = correct_titles.split(delimiter)
    lines3 = correct_categories.split(delimiter)
    
    for (l1, l2, l3, l4) in zip(lines1, lines2, lines3, links):
        s = u','.join([str(l1), str(l2), str(l3), str(l4)]) + u'\n'
        file.write(s)
        
print("AutoCorrection Finished!\n")