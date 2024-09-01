import spacy
from PyPDF2 import PdfReader
import pandas as pd

from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc

import pandas as pd
import numpy as np
import jsonlines

import re
from typing import List

from sentence_transformers import SentenceTransformer, util

import constants
from feature_extractor import FeaturesExtractor

class ResumeRater:
    
    def __init__(self):
        
        self._pretrained_model = SentenceTransformer(constants.PRETRAINED_SENTENCE_TRANSFORMERS_MODEL)
        
        self._RATING_WEIGHTS = constants.RATING_WEIGHTS
        self._YEARS_EXPERIENCES_WEIGHTS = constants.YEARS_EXPERIENCES_WEIGHTS
            
    def fit_transform(self, job_description_text: List[str], resume_text: List[dict]):
        
        self._extractor = FeaturesExtractor()
        
        job_text, job_features = self._extractor.fit_transform(job_description_text)

        # Only 1 job posting description
        self._job_text = job_text[0] 
        self._job_features = job_features[0]
        
        self._resume_text, self._resume_features = self._extractor.fit_transform(resume_text)
        
        for feature in self._resume_features:
        
            feature['rating_details'] = {
                
                'educations': self._rate_educations(feature['educations']),
                'gpa': self._rate_gpa(feature['gpa']),
                'job_titles' : self._rate_job_titles(feature['job_titles']),
                'years_experiences': self._rate_years_experiences(feature['years_experiences']),
                'experiences': self._rate_experiences(feature['experiences']),
                'skills': self._rate_skills(feature['skills']),
                'soft_skills': self._rate_soft_skills(feature['soft_skills']),
                'languages': self._rate_languages(feature['languages']),
                
            }
            
            final_rating = 0
            
            for category, rating in feature['rating_details'].items():
                
                try:
                    
                    calculated_rating =  rating * self._RATING_WEIGHTS[str(category).upper()]
                    
                except Exception as e:
                    
                    print(f'Error on parsing rating weights: {e}')
                    continue
                
                final_rating += calculated_rating
                 
            feature['rating'] = final_rating
            
        
        return self._job_features, self._resume_features
    
    def top_resume_by_rating(self, k = 5):
        
        if k <= 0:
            raise ValueError(f"Parameter k must be non-negative and non-zero, got {k}.")
        
        sorted_resumes = sorted(self._resume_features, key=lambda x: x['rating'], reverse=True)
    
        top_k_resumes = sorted_resumes[:k]

        return top_k_resumes
    
    def _check_feature_availability(self, job_feature: List[str], resume_feature: List[str]):
        
        if len(job_feature) <= 0 or not job_feature:
            return 1
        
        if len(resume_feature) <= 0 or not resume_feature:
            return 0
        
        return -1
    
    def _calculate_cosine_similarity_matrix_mean(self, job_feature_category: List[str], resume_feature_category: List[str], return_matrix = False, use_threshold= False, threshold= 0.42):
        
        check_feature = self._check_feature_availability(job_feature_category, resume_feature_category)
        
        if check_feature != -1:
            
            m = len(job_feature_category)
            n = len(resume_feature_category)
            
            return (check_feature, np.full((m, n), check_feature)) if return_matrix else check_feature
        
        job_embeddings = self._pretrained_model.encode(job_feature_category)
        resume_embeddings  = self._pretrained_model.encode(resume_feature_category)
        
        cosine_sim_matrix = util.pytorch_cos_sim(job_embeddings, resume_embeddings).numpy()
        
        top_similarity = cosine_sim_matrix.max(axis= 1) # Rows: Job, Columns: Resume -> Get max similarity per job
        
        score = top_similarity.mean()
        
        if use_threshold:

            matches = np.sum(top_similarity >= threshold)
                
            # Calculate the score based on the number of matches and similarity >= threshold
            if matches > 0:
                score = (matches / len(job_feature_category)) * np.mean(top_similarity[top_similarity >= threshold])
            else:
                score = 0.0
                
        if return_matrix:
            return float(score), cosine_sim_matrix
        
        return float(score)

    def _calculate_matching_words_score(self, job_word_list: List[str], resume_word_list: List[str]):
                
        check_feature = self._check_feature_availability(job_word_list, resume_word_list)
        
        if check_feature != -1:
            return check_feature
        
        job_word_list = [word.lower() for word in job_word_list]
        resume_word_list = [word.lower() for word in resume_word_list]
        
        score = 0
        
        for resume_word in resume_word_list:
            
            if resume_word in job_word_list:
                score += 1

        n_job_word = len(job_word_list)

        score /= n_job_word
        
        return score
    
    def _calculate_year_weight(self, job_year_list: List[int], resume_year_list: List[int]):
        
        check_feature = self._check_feature_availability(job_year_list, resume_year_list)
        
        if check_feature != -1:
            return check_feature
        
        m = len(job_year_list)
        n = len(resume_year_list)
        
        weight_mtx = np.zeros((m, n))
        
        for job_idx, job_year in enumerate(job_year_list):
            
            for resume_idx, resume_year in enumerate(resume_year_list):
                
                weight = min(resume_year / job_year, 1)
                            
                weight_mtx[job_idx, resume_idx] = weight
        
        return weight_mtx

    
    def _rate_educations(self, resume_feature: List[str]):
        
        return self._calculate_cosine_similarity_matrix_mean(self._job_features['educations'], resume_feature)
    
    def _rate_gpa(self, resume_feature: List[str]):
        
        return 0
    
    def _rate_job_titles(self, resume_feature: List[str]):
        
        return self._calculate_cosine_similarity_matrix_mean(self._job_features['job_titles'], resume_feature, use_threshold=False)
    
    def _rate_years_experiences(self, resume_feature: List[str]):
                
        job_keywords_context = [' '.join(features['keywords_context']) for features in self._job_features['years_experiences']]
        resume_keywords_context = [' '.join(features['keywords_context']) for features in resume_feature]

        if not job_keywords_context or job_keywords_context == ['']:
            similarity_mtx = np.array([[1]])
        elif not resume_keywords_context or job_keywords_context == ['']:
            similarity_mtx = np.array([[0]])
        else:
            _, similarity_mtx = self._calculate_cosine_similarity_matrix_mean(job_keywords_context, resume_keywords_context, return_matrix= True)
      
        job_year_list = [features['year'] for features in self._job_features['years_experiences']]
        resume_year_lsit = [features['year'] for features in resume_feature]
        
        year_weight_mtx = self._calculate_year_weight(job_year_list, resume_year_lsit)

        job_keywords_match = [features['keywords_match'] for features in self._job_features['years_experiences']]
        resume_keywords_match = [features['keywords_match'] for features in resume_feature]

        
        m = len(job_keywords_match)
        n = len(resume_keywords_match)
        
        if m <= 0:
            matching_mtx = np.array([[1]])
        elif n <= 0:
            matching_mtx = np.array([[0]])
        else:
            matching_mtx = np.zeros((m, n))

            for job_idx, job_match in enumerate(job_keywords_match):
                
                for resume_idx, resume_match in enumerate(resume_keywords_match):
                    
                    matching_mtx[job_idx, resume_idx] = self._calculate_matching_words_score(job_match, resume_match)

        
        
        weighted_similarity_mtx = similarity_mtx * self._YEARS_EXPERIENCES_WEIGHTS['KEYWORDS_CONTEXT']
        weighted_matching_mtx = matching_mtx * self._YEARS_EXPERIENCES_WEIGHTS['KEYWORDS_MATCH']
        
        final_mtx = (weighted_similarity_mtx + weighted_matching_mtx) * year_weight_mtx
        top_score = final_mtx.max(axis= 1)
        score = top_score.mean()
                
        return score
    
    def _rate_experiences(self, resume_feature: List[str]):
        
        return self._calculate_cosine_similarity_matrix_mean(self._job_features['experiences'], resume_feature)
        
    def _rate_skills(self, resume_feature: List[str]):
        
        return self._calculate_matching_words_score(self._job_features['skills'], resume_feature)
        
    def _rate_soft_skills(self, resume_feature: List[str]):
        
        return self._calculate_cosine_similarity_matrix_mean(self._job_features['soft_skills'], resume_feature)
        
    def _rate_languages(self, resume_feature: List[str]):
        
        return self._calculate_matching_words_score(self._job_features['languages'], resume_feature)
        
            
    