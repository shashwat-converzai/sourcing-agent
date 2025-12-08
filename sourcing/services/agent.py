from typing import List, Dict
from django.db import transaction

from ..models import (
    Job,
    CandidateJobApplication,
    JobCandidateMatch,
)
from .matching import get_candidate_pool_for_job, score_candidates_for_job


@transaction.atomic
def auto_source_for_job(
    job: Job, top_n: int = 20, similarity_threshold: float = 0.5
) -> List[Dict]:
    candidate_pool = get_candidate_pool_for_job(job)
    scored = score_candidates_for_job(job, candidate_pool)

    filtered = [s for s in scored if s["similarity"] >= similarity_threshold]
    top = filtered[:top_n]

    results = []

    for item in top:
        cand = item["candidate"]
        sim = item["similarity"]

        app, created = CandidateJobApplication.objects.get_or_create(
            candidate=cand,
            job=job,
            defaults={"status": CandidateJobApplication.RECOMMENDED},
        )
        if not created and app.status in [
            CandidateJobApplication.APPLIED,
            CandidateJobApplication.INTERESTED,
        ]:
            app.status = CandidateJobApplication.RECOMMENDED
            app.save(update_fields=["status", "last_status_change_at"])

        JobCandidateMatch.objects.update_or_create(
            job=job,
            candidate=cand,
            defaults={
                "score": sim,
                "semantic_match_score": sim,
            },
        )

        results.append({
            "candidate_id": cand.candidate_id,
            "full_name": cand.full_name,
            "similarity": sim,
        })

    return results
