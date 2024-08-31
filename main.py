from resume_rater import ResumeRater
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import List
import re

app = FastAPI()

origins = [
    "http://localhost:6969",
    "localhost:6969"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/rate")
async def rate_resume(job_description_text: List[str], resume_text: List[str]):
    
    resume_rater = ResumeRater()

    job_features, resume_features = resume_rater.fit_transform(job_description_text, resume_text)
    
    resume_features = sorted(resume_features, key=lambda x: x['rating'], reverse=True)

    return job_features, resume_features


@app.post("/rate-top")
async def rate_resume(job_description_text: List[str], resume_text: List[str], top_k: int = 5):
    
    resume_rater = ResumeRater()

    resume_text, resume_features = resume_rater.fit_transform(job_description_text, resume_text)
    
    top_k_resumes = resume_rater.top_resume_by_rating(top_k)
    
    return top_k_resumes
    