from resume_rater import ResumeRater
from fastapi import FastAPI
from typing import List
import re

app = FastAPI()


@app.post("/rate")
async def rate_resume(job_description_text: List[str], resume_text: List[str]):
    
    resume_rater = ResumeRater()

    resume_text, resume_features = resume_rater.fit_transform(job_description_text, resume_text)
    
    return resume_features


@app.post("/multiple-rate")
async def rate_resume(job_description_text: List[str], resume_text: List[str], top_k: int = 5):
    
    resume_rater = ResumeRater()

    resume_text, resume_features = resume_rater.fit_transform(job_description_text, resume_text)
    
    top_k_resumes = resume_rater.top_resume_by_rating(top_k)
    
    return top_k_resumes
    