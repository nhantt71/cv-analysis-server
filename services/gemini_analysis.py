import os
import re
import json
import redis
import logging
from datetime import datetime
import concurrent.futures
from dotenv import load_dotenv

from weasyprint import HTML
from pdf2image import convert_from_path
import base64
from io import BytesIO

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from pydantic import BaseModel

from services.database import (
    save_nli_analysis,
    save_cv_job_matches,
    get_all_jobs,
    get_cv_for_filter,
    get_relative_jobs
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API and Redis
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    db=int(os.getenv("REDIS_DB"))
)

os.environ['FONTCONFIG_PATH'] = r"C:\Program Files\GTK3-Runtime Win64\etc\fonts"


def extract_score(text: str) -> float:
    try:
        score_line = [line for line in text.splitlines() if "score" in line.lower()][-1]
        match = re.search(r"([0-1](\.\d+)?)", score_line)
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0


# ========== NLI Analysis ==========
def analyze_single_job(cv_id, cv_text, job):
    prompt_template = PromptTemplate.from_template("""
    You are a professional AI recruitment assistant.

    Your task is to compare the following CV and Job Description, and assess how well the CV matches the job.

    IMPORTANT:
    - Your evaluation must be based solely on the candidate's **skills, experiences, education, and relevant qualifications**.
    - You must **not** consider or mention any factors related to **gender, age, race, ethnicity, religion, marital status, physical appearance, political views**,
    or any **personally identifiable or protected attributes**.
    - Your assessment must be **fair, lawful, and free from bias or discrimination**.
    - Do not include any language that may violate **equal opportunity employment principles** or local/national labor laws.

    Return only a JSON response in this exact format:
    {{
      "score": <float between 0 and 1>,
      "explanation": "<why this CV does or does not match>"
    }}

    CV:
    {cv_text}

    Job Description:
    {job_detail}
    """)

    chain = prompt_template | llm  # ensure `llm` is initialized

    try:
        response = chain.invoke({
            "cv_text": cv_text,
            "job_detail": job['detail']
        })

        raw_content = response.content if hasattr(response, "content") else str(response)
        cleaned_content = raw_content.strip().strip("`")

        match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in Gemini response")

        json_str = match.group()
        parsed = json.loads(json_str)

        score = float(parsed.get("score", 0))
        explanation = parsed.get("explanation", "")

        if score >= 0.5:
            save_cv_job_matches(cv_id, job["id"], cv_text, score, explanation)
            return {
                "job_id": job["id"],
                "match_score": score,
                "explanation": explanation,
                "created_at": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error analyzing job {job['id']}: {e}")
        return None


def analyze_cv_with_jobs(cv_text: str, cv_id: int):
    cache_key = f"nli_analysis:cv:{cv_id}"
    cache_key2 = f"recommend:cv:{cv_id}"

    if r.exists(cache_key):
        return json.loads(r.get(cache_key))

    jobs = get_all_jobs()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(analyze_single_job, cv_id, cv_text, job)
            for job in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    sorted_results = sorted(results, key=lambda x: x["match_score"], reverse=True)

    save_nli_analysis(cv_id, sorted_results)
    r.set(cache_key, json.dumps(sorted_results), ex=1800)  # Cache 30 minutes
    r.set(cache_key2, json.dumps(sorted_results), ex=1800) # Cache 30 minutes
    return sorted_results


# ========== Candidate Filtering ==========

class FilterRequest(BaseModel):
    filters: dict


def filter_candidates(req: FilterRequest):
    rows = get_cv_for_filter()
    filters_str = json.dumps(req.filters, ensure_ascii=False, indent=2)

    candidate_filter_prompt = PromptTemplate.from_template("""
    You are an AI recruitment assistant. Below are the filtering criteria provided by the recruiter:

    {filters}

    Here is the CV content of a candidate:

    {cv_text}

    Your task:
    - Evaluate how well this CV matches the filtering criteria.
    - Your evaluation must be based **only on relevant skills, experiences, education, certifications, and job-related qualifications**.
    - You must **NOT** consider or mention any information related to **gender, age, race, ethnicity, religion, political belief, marital status,
    or any other protected personal attributes**.
    - Do **not** make assumptions if data is missing — just indicate it as insufficient information.
    - Your response must be objective, fair, and in compliance with anti-discrimination and labor laws.

    Return only a JSON response in the following format:
    {{
      "match_score": 0.xx,
      "reason": "Short explanation why this CV is or is not a good match."
    }}

    The match_score is a float value between 0 and 1.

    Be accurate. If the CV lacks key information or doesn't match the criteria, give a low score and clearly explain why.
    """)

    chain = candidate_filter_prompt | llm

    best_scores = {}  # key = cv_id

    for cv_id, cv_text in rows:
        try:
            response = chain.invoke({"filters": filters_str, "cv_text": cv_text})
            response_text = response.content if hasattr(response, "content") else str(response)
            response_text = response_text.strip().strip("`")

            if not response_text:
                raise ValueError("Empty response from Gemini")

            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                raise ValueError("No valid JSON found in Gemini response")

            json_str = match.group()
            parsed = json.loads(json_str)
            match_score = parsed.get("match_score", 0)
            reason = parsed.get("reason", "")

            if match_score >= 0.6:
                if cv_id not in best_scores or match_score > best_scores[cv_id]["match_score"]:
                    best_scores[cv_id] = {
                        "cv_id": cv_id,
                        "match_score": match_score,
                        "reason": reason
                    }

        except Exception as e:
            logger.error(f"Error while processing CV {cv_id}: {e}")

    return {
        "matched_candidates": sorted(best_scores.values(), key=lambda x: -x["match_score"])
    }


# ========== Related Jobs ==========

def related_jobs(job_id: int):
    cache_key = f"related_jobs:{job_id}"
    if r.exists(cache_key):
        return json.loads(r.get(cache_key))

    other_jobs, job = get_relative_jobs(job_id)
    results = []

    prompt_template = PromptTemplate.from_template("""
Compare the similarity between the following two jobs:

Job 1:
{target_text}

Job 2:
{compare_text}

Return a similarity score between 0 and 1 (as a float), and a short explanation of the reasoning.

Respond in JSON format:
{{"score": 0.0, "explanation": "..."}}
""")

    chain = prompt_template | llm
    target_text = f"Job Name: {job.name}\nJob Description: {job.detail}"

    def analyze_job(other):
        compare_text = f"Job Name: {other.name}\nJob Description: {other.detail}"
        try:
            response = chain.invoke({
                "target_text": target_text,
                "compare_text": compare_text
            })
            raw_content = response.content if hasattr(response, "content") else str(response)
            cleaned_content = raw_content.strip().strip("`")
            if cleaned_content.startswith("json"):
                cleaned_content = cleaned_content[4:].strip()

            parsed = json.loads(cleaned_content)
            score = float(parsed.get("score", 0))
            explanation = parsed.get("explanation", "")

            if score >= 0.4:
                return {
                    "jobId": other.id,
                    "score": round(score, 3),
                    "explanation": explanation
                }
        except Exception as e:
            logger.warning(f"Failed comparing with job {other.id}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_job, job) for job in other_jobs]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    r.setex(cache_key, 10800, json.dumps(results))  # Cache 6h

    return results


# llm2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=1)
#
#
# def generate_resume_text(cv_data: dict) -> str:
#     prompt_template = PromptTemplate.from_template("""
# You are a professional resume builder. Your task is to generate a beautiful, modern, single-page resume using the provided structured data.
#
# The output must be in **HTML** format and include **inline CSS styling** for formatting.
#
# 🔧 Requirements:
# - Font: Helvetica or Arial, size 11–12pt.
# - Sections: Name, Contact, Summary, Skills, Experience, Education, Certifications, Projects (if available).
# - Section titles should be bold and slightly larger.
# - Use clean spacing, consistent margins, and modern layout.
# - Preferably use a **two-column layout**: left for contact/skills/education, right for summary/experience/projects.
# - Limit content to one page’s worth (approx. 800–1000 words).
# - Do NOT include anything unrelated or artificial.
# - Use modern CSS styling including:
#    - Clean grid or flexbox layout
#    - Color scheme (suggest: #333 for text, #0056b3 for accent colors)
#    - Proper padding (15-20px) and margins
#    - Section dividers
#    - Professional typography with font-weight adjustments
#
# Here is the structured CV data (in JSON format):
#
# {cv_data}
#
# Now generate the complete HTML resume:
# """)
#     chain = prompt_template | llm2
#     response = chain.invoke({"cv_data": json.dumps(cv_data, ensure_ascii=False, indent=2)})
#
#     return getattr(response, "content", str(response)).strip()
#
#
# import tempfile
#
# def html_to_image_base64(html_content: str) -> str:
#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
#         HTML(string=html_content).write_pdf(tmp_pdf.name)
#         images = convert_from_path(tmp_pdf.name)
#
#     img = images[0]
#     buffered = BytesIO()
#     img.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")
#

