import numpy as np
from typing import Dict, Optional

from django.db import transaction

from ..models import (
    Candidate, Job,
    CandidateVector, JobVector,
    CandidateSkill,
)


def get_text_embedding(text: str) -> np.ndarray:
    """
    Stub: plug in your embedding provider here.
    Must return a 1D numpy array.
    """
    # For now, return a deterministic dummy vector based on length to keep things running.
    # Replace this with a real embedding model (e.g., OpenAI, HuggingFace).
    dim = 64
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim).astype(float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


INDUSTRY_INDEX: Dict[str, int] = {
    "it": 0,
    "logistics": 1,
    "healthcare": 2,
}

DOMAIN_INDEX: Dict[str, int] = {
    "software": 0,
    "hr": 1,
    "finance": 2,
    "accounts": 3,
    "business": 4,
    "admin": 5,
    "doctor": 6,
    "nurse": 7,
    "plumber": 8,
    "mechanic": 9,
}

SENIORITY_INDEX: Dict[str, float] = {
    "junior": 0.2,
    "mid-level": 0.5,
    "mid-senior": 0.7,
    "senior": 1.0,
}


def one_hot(index_map: Dict[str, int], value: Optional[str]) -> np.ndarray:
    dim = len(index_map)
    vec = np.zeros(dim, dtype=float)
    if not value:
        return vec
    key = value.lower()
    if key in index_map:
        vec[index_map[key]] = 1.0
    return vec


def encode_seniority(value: Optional[str]) -> float:
    if not value:
        return 0.5
    return SENIORITY_INDEX.get(value.lower(), 0.5)


def build_candidate_structured_vector(candidate: Candidate) -> np.ndarray:
    industry_vec = one_hot(INDUSTRY_INDEX, candidate.industry)
    domain_vec = one_hot(DOMAIN_INDEX, candidate.domain)
    seniority_val = encode_seniority(candidate.seniority_level)

    total_exp = candidate.total_experience_years or 0.0
    exp_norm = total_exp / 40.0
    exp_features = np.array([exp_norm, exp_norm], dtype=float)  # ← 2 dims now

    skills_vec = np.zeros(32, dtype=float)
    for cs in candidate.candidate_skills.select_related("skill").all():
        name = cs.skill.name.lower()
        idx = (ord(name[0]) % 32) if name else 0
        skills_vec[idx] += 1.0

    structured = np.concatenate([
        industry_vec,
        domain_vec,
        np.array([seniority_val], dtype=float),
        exp_features,
        skills_vec,
    ])
    return structured


def build_candidate_text(candidate: Candidate) -> str:
    parts = [
        candidate.current_designation or "",
        candidate.present_location or "",
        candidate.permanent_location or "",
    ]

    skills = candidate.candidate_skills.select_related("skill").all()
    skill_names = ", ".join(cs.skill.name for cs in skills)
    parts.append(skill_names)

    for exp in candidate.experiences.all():
        snippet = f"{exp.designation} at {exp.company_name}. {exp.responsibilities or ''}"
        parts.append(snippet)

    if candidate.resume_raw:
        parts.append(candidate.resume_raw[:4000])

    return "\n".join(parts)


def build_job_structured_from_meta(meta: dict) -> np.ndarray:
    """
    Build the structured part of a job vector from a plain dict (no DB Job).
    """
    industry = meta.get("industry")
    domain = meta.get("domain")
    seniority = meta.get("seniority_level")

    industry_vec = one_hot(INDUSTRY_INDEX, industry)
    domain_vec = one_hot(DOMAIN_INDEX, domain)
    seniority_val = encode_seniority(seniority)

    exp_min = meta.get("experience_min_years") or 0.0
    exp_max = meta.get("experience_max_years") or exp_min
    exp_features = np.array([exp_min / 40.0, exp_max / 40.0], dtype=float)

    # required_skills is a list of skill names (strings)
    skills_vec = np.zeros(32, dtype=float)
    for name in meta.get("required_skills", []) or []:
        name = name.lower()
        idx = (ord(name[0]) % 32) if name else 0
        skills_vec[idx] += 1.0

    structured = np.concatenate([
        industry_vec,
        domain_vec,
        np.array([seniority_val], dtype=float),
        exp_features,
        skills_vec,
    ])
    return structured


def build_job_text_from_meta(meta: dict) -> str:
    """
    Build textual representation for embeddings from a job metadata dict.
    """
    parts = [
        meta.get("title") or "",
        meta.get("location") or "",
        meta.get("description") or "",
        f"Industry: {meta.get('industry') or ''}",
        f"Domain: {meta.get('domain') or ''}",
        f"Seniority: {meta.get('seniority_level') or ''}",
    ]

    for sk in meta.get("required_skills", []) or []:
        parts.append(f"must-have skill: {sk}")

    return "\n".join(parts)


def build_job_vector_from_meta(meta: dict) -> np.ndarray:
    """
    Full job vector (semantic + structured) from metadata (no DB Job).
    Must match the dimension of JobVector.vector.
    """
    text = build_job_text_from_meta(meta)
    semantic_vec = get_text_embedding(text)
    structured_vec = build_job_structured_from_meta(meta)
    unified = np.concatenate([semantic_vec, structured_vec])
    return unified


def build_job_structured_vector(job: Job) -> np.ndarray:
    industry_vec = one_hot(INDUSTRY_INDEX, job.industry)
    domain_vec = one_hot(DOMAIN_INDEX, job.domain)
    seniority_val = encode_seniority(job.seniority_level)

    exp_min = job.experience_min_years or 0.0
    exp_max = job.experience_max_years or exp_min
    exp_features = np.array([exp_min / 40.0, exp_max / 40.0], dtype=float)

    skills_vec = np.zeros(32, dtype=float)
    for js in job.job_skills.select_related("skill").all():
        name = js.skill.name.lower()
        idx = (ord(name[0]) % 32) if name else 0
        skills_vec[idx] += 1.0

    structured = np.concatenate([
        industry_vec,
        domain_vec,
        np.array([seniority_val], dtype=float),
        exp_features,
        skills_vec,
    ])
    return structured


def build_job_text(job: Job) -> str:
    parts = [
        job.title or "",
        job.location or "",
        job.description or "",
    ]
    for js in job.job_skills.select_related("skill").all():
        tag = "must-have" if js.importance == js.MUST_HAVE else "nice-to-have"
        parts.append(f"{tag} skill: {js.skill.name}")
    return "\n".join(parts)


@transaction.atomic
def generate_candidate_vector(candidate: Candidate) -> CandidateVector:
    text = build_candidate_text(candidate)
    semantic_vec = get_text_embedding(text)
    structured_vec = build_candidate_structured_vector(candidate)
    unified = np.concatenate([semantic_vec, structured_vec])

    cv, _ = CandidateVector.objects.update_or_create(
        candidate=candidate,
        defaults={
            "vector": unified.tolist(),
            "semantic_vector": semantic_vec.tolist(),
            "structured_vector": structured_vec.tolist(),
        },
    )
    return cv


@transaction.atomic
def generate_job_vector(job: Job) -> JobVector:
    text = build_job_text(job)
    semantic_vec = get_text_embedding(text)
    structured_vec = build_job_structured_vector(job)
    unified = np.concatenate([semantic_vec, structured_vec])

    jv, _ = JobVector.objects.update_or_create(
        job=job,
        defaults={
            "vector": unified.tolist(),
            "semantic_vector": semantic_vec.tolist(),
            "structured_vector": structured_vec.tolist(),
        },
    )
    return jv
