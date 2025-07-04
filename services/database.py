import json

import redis
import psycopg2
from dotenv import load_dotenv

from fastapi import HTTPException
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Boolean, Text, DateTime, func
)
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
import os

# Load environment variables
load_dotenv()

# === Database Configuration ===
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@" \
               f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === Redis Configuration ===
red = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=int(os.getenv("REDIS_DB"))
)

# === Gemini Model ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

# === SQLAlchemy Model ===
class Job(Base):
    __tablename__ = "job"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    detail = Column(Text)
    enable = Column(Boolean)
    end_date = Column(DateTime)
    experience = Column(String)

# === Helper: Connect PostgreSQL ===
def connect():
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD2")

    return psycopg2.connect(
        host=host,
        port=port,
        dbname=db,
        user=user,
        password=password
    )


# === Fetch Active Jobs ===
def get_all_jobs():
    with SessionLocal() as session:
        now = func.now()
        jobs = session.query(Job).filter(Job.end_date > now, Job.enable == True).all()
        return [
            {
                "id": job.id,
                "name": job.name,
                "enable": job.enable,
                "detail": job.detail,
                "experience": job.experience
            }
            for job in jobs
        ]

def get_relative_jobs(job_id: int):
    with SessionLocal() as session:
        job = session.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        now = func.now()
        other_jobs = session.query(Job).filter(
            Job.end_date > now,
            Job.enable == True,
            Job.id != job_id
        ).all()
    return other_jobs, job


# === Save NLI Analysis for a CV ===
def save_nli_analysis(cv_id: int, results: list):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO nli_analysis (cv_id, analysis) VALUES (%s, %s)",
        (cv_id, json.dumps(results))
    )
    conn.commit()
    cur.close()
    conn.close()

# === Save a Match between CV and Job ===
def save_cv_job_matches(cv_id: int, job_id: int, cv_text: str, score: float, explanation: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO cv_job_matches (cv_id, job_id, cv_text, match_score, explanation)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (cv_id, job_id, cv_text, score, explanation)
    )
    conn.commit()
    cur.close()
    conn.close()


# === Recommend Jobs for a CV ===
def recommend_jobs_for_cv(cv_id: int):
    cache_key = f"recommend:cv:{cv_id}"
    if red.exists(cache_key):
        return red.get(cache_key).decode()

    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, match_score, explanation
        FROM cv_job_matches
        WHERE cv_id = %s
        ORDER BY match_score DESC
        LIMIT 10
        """,
        (cv_id,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = [{"job_id": r[0], "score": float(r[1]), "explanation": r[2]} for r in rows]
    red.set(cache_key, json.dumps(result))
    return json.dumps(result)

# === Recommend CVs for a Job ===
def recommend_cvs_for_job(job_id: int):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cv_id, match_score, explanation
        FROM cv_job_matches
        WHERE job_id = %s
        ORDER BY match_score DESC
        LIMIT 10
        """,
        (job_id,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = [{"cv_id": r[0], "score": float(r[1]), "explanation": r[2]} for r in rows]
    return json.dumps(result)

# === Get all CVs and Text for Filtering ===
def get_cv_for_filter():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT cv_id, cv_text FROM cv_job_matches")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# === AI Agent Tools ===
tools = [
    Tool(
        name="RecommendJobsForCV",
        func=lambda q: recommend_jobs_for_cv(int(q)),
        description="Given a CV ID, return top 10 matching job IDs based on match score."
    ),
    Tool(
        name="RecommendCVsForJob",
        func=lambda q: recommend_cvs_for_job(int(q)),
        description="Given a Job ID, return top 10 matching CVs based on match score."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
