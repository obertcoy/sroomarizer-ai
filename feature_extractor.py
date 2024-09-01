import spacy
import pandas as pd

from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc

import pandas as pd
import numpy as np
import jsonlines

import re
from typing import List
import constants

class FeaturesExtractor:

    def __init__(self):

        self.nlp = spacy.load('./data/train_model/model_ner/')
        
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        
        if 'entity_ruler' not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe('entity_ruler', after='ner')
            ruler.from_disk(constants.SKILLS_PATTERN_PATH)
            
            job_title_patterns = pd.read_csv(constants.JOB_TITLE_PATH)['Job Title'].unique()
            
            for title in job_title_patterns:
                ruler.add_patterns([{"label": "JOB TITLE", "pattern": title}])
        
        self._CATEGORIES_PATTERN = constants.CATEGORIES_PATTERN
    
    def fit_transform(self, input: List[str]):
        
        self._text_arr = []
        self._feature_arr = []
        self._input_len = len(input)

        for i in range(self._input_len):
            
            doc = self._remove_excess_spaces(input[i])
            self._text_arr.append(doc.text)

            self._extract_features(i, doc)

        return self._text_arr, self._feature_arr

    def _remove_excess_spaces(self, text):
            
        doc = self.nlp(re.sub(r'\s+', ' ', text).strip())

        return doc
            

    def _extract_features(self, resume_idx, doc):

        feature_dict = {

            'resume_idx': resume_idx,
            'name': self._extract_name(doc),
            'phone': self._extract_phone(doc),
            'educations': self._extract_educations(doc),
            'gpa': self._extract_gpa(doc),
            'job_titles' : self._extract_job_titles(doc),
            'years_experiences': self._extract_years_experiences(doc),
            'experiences': self._extract_experiences(doc),
            'skills': self._extract_skills(doc),
            'soft_skills': self._extract_soft_skills(doc),
            'languages': self._extract_languages(doc),
            
        }
        
        self._feature_arr.append(feature_dict)
           

    
    def _extract_name(self, doc):
        
        name = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name.append(ent.text)

        return name


    def _extract_phone(self, doc):

        pattern = r'(?:\+?(?:\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?(?:\d{2,4}[-.\s]?){2,5}\d{2,4})'

        matches = re.findall(pattern, doc.text)

        return matches
    
    def _extract_educations(self, doc):
        
        educations = []

        pattern = self._CATEGORIES_PATTERN['EDUCATIONS']
        
        matches = re.findall(pattern, doc.text)
        for match in matches:
            educations.append(match.strip())
            
        for ent in doc.ents:
            if 'DIPLOMA' in ent.label_:
                educations.append(ent.text)

        return [edu for edu in set(educations)]
    
    def _extract_gpa(self, doc):
        
        gpas = []
        
        for ent in doc.ents:
            if 'GPA' in ent.label_:
                gpas.append(ent.text)
                
        return [gpa.capitalize() for gpa in set(gpas)]
    
    def _extract_job_titles(self, doc):
        
        job_titles = []
        
        for ent in doc.ents:
            if 'JOB TITLE' in ent.label_:
                job_titles.append(ent.text)
        
        return [job for job in set(job_titles)]
    
    def _extract_years_experiences(self, doc):
        
        years_experiences= []
        
        pattern = self._CATEGORIES_PATTERN['YEARS_EXPERIENCES']
        
        sentences = [sent.text.strip() for sent in doc.sents]

        # for sentence in sentences:
        #     if re.search(pattern, sentence, re.IGNORECASE):
        #         years_experiences_sentences.append(sentence)
        
        for sentence in sentences:
            
            matches = re.findall(pattern, sentence, re.IGNORECASE)
                        
            for match in matches:
                
                # Prevent context not parsed
                try:

                    
                    year = match  # Extracted years
                    year = re.sub(r'\+', '', year)  # Remove '+' if present

                    try:
                        
                        year = int(year)
                    
                    except Exception as e:
                        
                        print(f"Error converting {match} to number: {e}")
                        continue

                    match_doc = self.nlp(sentence)
                    
                    skills = self._extract_skills(match_doc)
                    job_titles = self._extract_job_titles(match_doc)
                    languages = self._extract_languages(match_doc)
                    
                    # Make string out of keywords for similarity scoring
                    keywords_match = ' '.join(skills + languages)
                    keywords_context = ' '.join(job_titles)

                    if keywords_match or keywords_context:

                        years_experience_dict = {
                            'text': sentence,
                            'year': year,
                            'keywords_match': keywords_match,
                            'keywords_context': keywords_context
                        }
                        
                        years_experiences.append(years_experience_dict)

                except Exception as e:
                    print(f"Error processing match {match}: {e}")
                    continue 
            
        return years_experiences
    
    def _extract_experiences(self, doc):
        
        experiences = []
        
        pattern = self._CATEGORIES_PATTERN['EXPERIENCES']
        
        sentences = [sent.text.strip() for sent in doc.sents]

        for sentence in sentences:
            if re.search(pattern, sentence, re.IGNORECASE):
                experiences.append(sentence)
                
        for ent in doc.ents:
            if 'EXPERIENCE' in ent.label_:
                experiences.append(ent.text)
        
        return [exp for exp in set(experiences)]
    
    def _extract_skills(self, doc):

        skills = []

        for ent in doc.ents:
            if 'SKILL' in ent.label_ and 'SOFT SKILL' not in ent.label_:
                skills.append(ent.text)
                
        return [skill.capitalize() for skill in set(skills)]
    
    def _extract_soft_skills(self, doc):
        
        soft_skills = []
        
        for ent in doc.ents:
            if 'SOFT SKILL' in ent.label_:
                soft_skills.append(ent.text)
                
        return [soft_skill for soft_skill in set(soft_skills)]

    def _extract_languages(self, doc):
        
        languages = []

        for ent in doc.ents:
            if 'LANGUAGE' in ent.label_:
                languages.append(ent.text)
                
        return [language.capitalize() for language in set(languages)]

    


