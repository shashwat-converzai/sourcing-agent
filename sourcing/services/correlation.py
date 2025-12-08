import numpy as np
from django.db import transaction

from ..models import CandidateCluster, JobCluster, ClusterCorrelation
from .vectorizer import cosine_similarity


def compute_cluster_correlations():
    job_clusters = list(JobCluster.objects.all())
    cand_clusters = list(CandidateCluster.objects.all())
    if not job_clusters or not cand_clusters:
        return

    with transaction.atomic():
        ClusterCorrelation.objects.all().delete()

        for jc in job_clusters:
            jc_vec = np.array(jc.center_vector, dtype=float)
            for cc in cand_clusters:
                cc_vec = np.array(cc.center_vector, dtype=float)
                corr = cosine_similarity(jc_vec, cc_vec)
                ClusterCorrelation.objects.create(
                    job_cluster=jc,
                    candidate_cluster=cc,
                    correlation=corr,
                )
