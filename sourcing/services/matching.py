import numpy as np
from typing import List, Dict

from ..models import (
    Job, Candidate,
    CandidateVector,
    CandidateClusterMembership, ClusterCorrelation,
    JobCluster,
)
from .vectorizer import cosine_similarity, build_job_vector_from_meta


def match_candidates_for_job_meta(
    job_meta: dict,
    top_n: int = 20,
    similarity_threshold: float = 0.5,
    top_k_clusters: int = 5,
    min_corr: float = 0.3,
    max_candidates: int = 2000,
) -> List[Dict]:
    """
    1) Build vector from job_meta
    2) Find nearest JobCluster
    3) Use ClusterCorrelation to get CandidateClusters
    4) Collect candidates in those clusters
    5) Score candidates by cosine similarity with job_meta vector
    6) Return top N
    """
    # 1. build job_meta vector
    job_vec = build_job_vector_from_meta(job_meta)
    job_vec_np = np.array(job_vec, dtype=float)

    # 2. find best job cluster
    best_jc = find_best_job_cluster_for_meta(job_vec_np)
    if best_jc is None:
        return []

    # 3. candidate pool from correlated clusters
    candidate_pool = get_candidate_pool_for_job_cluster(
        best_jc,
        top_k_clusters=top_k_clusters,
        min_corr=min_corr,
        max_candidates=max_candidates,
    )

    if not candidate_pool:
        return []

    # 4. load candidate vectors
    cvs = CandidateVector.objects.filter(
        candidate__in=candidate_pool).select_related("candidate")
    cv_map = {cv.candidate_id: cv for cv in cvs}

    scored: List[Dict] = []
    for cand in candidate_pool:
        cv = cv_map.get(cand.id)
        if not cv:
            continue
        cand_vec = np.array(cv.vector, dtype=float)
        sim = cosine_similarity(job_vec_np, cand_vec)
        if sim >= similarity_threshold:
            scored.append({
                "candidate": cand,
                "similarity": sim,
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:top_n]

    results = []
    for item in top:
        cand = item["candidate"]
        results.append({
            "candidate_id": cand.candidate_id,
            "full_name": cand.full_name,
            "similarity": item["similarity"],
            "present_location": cand.present_location,
            "domain": cand.domain,
            "industry": cand.industry,
            "seniority_level": cand.seniority_level,
        })

    return results


def get_candidate_pool_for_job_cluster(
    job_cluster: JobCluster,
    top_k_clusters: int = 5,
    min_corr: float = 0.3,
    max_candidates: int = 2000,
) -> List[Candidate]:
    """
    Use ClusterCorrelation to find candidate clusters correlated with a given job cluster,
    then gather candidates from those clusters.
    """
    correlations = (
        ClusterCorrelation.objects
        .filter(job_cluster=job_cluster, correlation__gte=min_corr)
        .order_by("-correlation")[:top_k_clusters]
    )

    if not correlations:
        # Fallback: all candidates (or you could return empty)
        return list(Candidate.objects.all()[:max_candidates])

    cluster_ids = [c.candidate_cluster_id for c in correlations]

    memberships = (
        CandidateClusterMembership.objects
        .filter(cluster_id__in=cluster_ids)
        .select_related("candidate")
        .order_by("distance_to_center")[:max_candidates]
    )

    return [m.candidate for m in memberships]


def find_best_job_cluster_for_meta(job_vec: np.ndarray) -> JobCluster | None:
    """
    Given an ad-hoc job vector, find the closest JobCluster by cosine similarity.
    """
    clusters = list(JobCluster.objects.all())
    if not clusters:
        return None

    best_cluster = None
    best_sim = -1.0

    for jc in clusters:
        jc_vec = np.array(jc.center_vector, dtype=float)
        sim = cosine_similarity(job_vec, jc_vec)
        if sim > best_sim:
            best_sim = sim
            best_cluster = jc

    return best_cluster


def get_relevant_candidate_clusters_for_job(job: Job, top_k: int = 5, min_corr: float = 0.3):
    j_membership = getattr(job, "cluster_membership", None)
    if not j_membership:
        return []
    jc = j_membership.cluster
    correlations = (
        ClusterCorrelation.objects
        .filter(job_cluster=jc, correlation__gte=min_corr)
        .order_by("-correlation")[:top_k]
    )
    return correlations


def get_candidate_pool_for_job(job: Job, max_candidates: int = 2000) -> List[Candidate]:
    correlations = get_relevant_candidate_clusters_for_job(job)
    if not correlations:
        return list(Candidate.objects.all()[:max_candidates])

    cluster_ids = [c.candidate_cluster_id for c in correlations]

    memberships = (
        CandidateClusterMembership.objects
        .filter(cluster_id__in=cluster_ids)
        .select_related("candidate")
        .order_by("distance_to_center")[:max_candidates]
    )

    return [m.candidate for m in memberships]


def score_candidates_for_job(job: Job, candidates: List[Candidate]) -> List[Dict]:
    jv = getattr(job, "vector", None)
    if not jv:
        raise ValueError("Job has no vector; generate vectors first.")

    job_vec = np.array(jv.vector, dtype=float)

    cvs = CandidateVector.objects.filter(
        candidate__in=candidates).select_related("candidate")
    cv_map = {cv.candidate_id: cv for cv in cvs}

    results = []
    for cand in candidates:
        cv = cv_map.get(cand.id)
        if not cv:
            continue
        cand_vec = np.array(cv.vector, dtype=float)
        sim = cosine_similarity(job_vec, cand_vec)
        results.append({
            "candidate": cand,
            "similarity": sim,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
