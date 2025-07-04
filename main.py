from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from services.database import recommend_jobs_for_cv, recommend_cvs_for_job
from services.gemini_analysis import FilterRequest, filter_candidates, related_jobs, analyze_cv_with_jobs
    # generate_resume_text, html_to_image_base64


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/recommend/jobs-for-cv/{cv_id}")
def get_job_recommendations(cv_id: int):
    result = recommend_jobs_for_cv(cv_id)
    return {"cv_id": cv_id, "recommended_jobs": result}

@app.get("/api/recommend/cvs-for-job/{job_id}")
def get_job_recommendations(job_id: int):
    result = recommend_cvs_for_job(job_id)
    return {"job_id": job_id, "recommended_cvs": result}

@app.post("/api/filter")
def get_filter_cvs(req: FilterRequest):
    return filter_candidates(req)

@app.get("/api/related-jobs/{job_id}")
def get_related_jobs(job_id: int):
    return related_jobs(job_id)

class CVBody(BaseModel):
    cv_text: str
    cv_id: int

@app.post("/api/analyze/cv")
def analyze_cv(req: CVBody):
    try:
        results = analyze_cv_with_jobs(req.cv_text, req.cv_id)
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error during analysis: {str(e)}"}
        )


class ResumeRequest(BaseModel):
    cv_data: dict


# @app.post("/api/resume-builder")
# def build_resume(request: ResumeRequest):
#     try:
#         resume_html = generate_resume_text(request.cv_data)
#         return {
#             "status": "success",
#             "resume_html": resume_html
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


import redis
import os

from dotenv import load_dotenv

load_dotenv()


redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    db=int(os.getenv("REDIS_DB"))
)


@app.get("/api/cache/check-exists")
async def check_cache_exists(key: str):
    """
    Check if a cache key exists in Redis
    """
    try:
        exists = redis_client.exists(key)
        return exists == 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check cache: {str(e)}")


@app.delete("/api/cache/delete")
async def delete_cache(key: str):
    """
    Delete a cache key from Redis
    """
    try:
        deleted = redis_client.delete(key)
        if deleted == 0:
            return {"success": False, "message": "Key not found"}
        return {"success": True, "message": "Cache deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cache: {str(e)}")


@app.get("/api/cache/list-keys")
async def list_cache_keys(pattern: str = "*"):
    """
    List all cache keys matching a pattern
    """
    try:
        keys = redis_client.keys(pattern)
        return {"keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cache keys: {str(e)}")


@app.get("/api/cache/get-ttl")
async def get_cache_ttl(key: str):
    """
    Get the TTL (time to live) for a cache key
    """
    try:
        ttl = redis_client.ttl(key)
        if ttl == -2:
            return {"ttl": None, "exists": False, "message": "Key does not exist"}
        elif ttl == -1:
            return {"ttl": None, "exists": True, "message": "Key has no expiration"}
        return {"ttl": ttl, "exists": True, "message": f"Key expires in {ttl} seconds"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get TTL: {str(e)}")
