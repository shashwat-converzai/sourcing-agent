import numpy as np
from sklearn.cluster import KMeans

from django.db import transaction

from ..models import (
    CandidateVector, JobVector,
    CandidateCluster, JobCluster,
    CandidateClusterMembership, JobClusterMembership,
)


def cluster_candidates(n_clusters: int = 30, random_state: int = 42):
    vectors_qs = CandidateVector.objects.all()
    if not vectors_qs.exists():
        return

    X = np.array([np.array(cv.vector, dtype=float) for cv in vectors_qs])

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    with transaction.atomic():
        CandidateClusterMembership.objects.all().delete()
        CandidateCluster.objects.all().delete()

        clusters = []
        for i in range(n_clusters):
            center_vec = centers[i].tolist()
            cluster = CandidateCluster.objects.create(
                name=f"CandidateCluster-{i}",
                center_vector=center_vec,
                size=int((labels == i).sum()),
            )
            clusters.append(cluster)

        for cv, label in zip(vectors_qs, labels):
            cluster = clusters[label]
            dist = float(np.linalg.norm(np.array(cv.vector, dtype=float) - np.array(cluster.center_vector, dtype=float)))
            CandidateClusterMembership.objects.create(
                candidate=cv.candidate,
                cluster=cluster,
                distance_to_center=dist,
            )


def cluster_jobs(n_clusters: int = 20, random_state: int = 42):
    vectors_qs = JobVector.objects.all()
    if not vectors_qs.exists():
        return

    X = np.array([np.array(jv.vector, dtype=float) for jv in vectors_qs])

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    with transaction.atomic():
        JobClusterMembership.objects.all().delete()
        JobCluster.objects.all().delete()

        clusters = []
        for i in range(n_clusters):
            center_vec = centers[i].tolist()
            cluster = JobCluster.objects.create(
                name=f"JobCluster-{i}",
                center_vector=center_vec,
                size=int((labels == i).sum()),
            )
            clusters.append(cluster)

        for jv, label in zip(vectors_qs, labels):
            cluster = clusters[label]
            dist = float(np.linalg.norm(np.array(jv.vector, dtype=float) - np.array(cluster.center_vector, dtype=float)))
            JobClusterMembership.objects.create(
                job=jv.job,
                cluster=cluster,
                distance_to_center=dist,
            )
