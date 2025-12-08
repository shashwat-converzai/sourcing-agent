from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from django.shortcuts import get_object_or_404

from .models import Job
from .serializers import JobSerializer, CandidateSerializer, JobMetaSerializer
from .services.agent import auto_source_for_job
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from django.shortcuts import get_object_or_404
from .services.matching import match_candidates_for_job_meta


@api_view(["GET"])
def list_jobs(request):
    jobs = Job.objects.all().order_by("-created_at")[:100]
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def job_detail(request, job_id):
    job = get_object_or_404(Job, job_id=job_id)
    serializer = JobSerializer(job)
    return Response(serializer.data)


@api_view(["POST"])
def auto_source_job(request, job_id):
    job = get_object_or_404(Job, job_id=job_id)
    top_n = int(request.data.get("top_n", 20))
    similarity_threshold = float(request.data.get("similarity_threshold", 0.5))

    results = auto_source_for_job(
        job, top_n=top_n, similarity_threshold=similarity_threshold)

    # Only return candidate summaries
    payload = []
    for r in results:
        payload.append({
            "candidate_id": r["candidate_id"],
            "full_name": r["full_name"],
            "similarity": r["similarity"],
        })

    return Response(payload, status=status.HTTP_200_OK)


@api_view(["POST"])
def match_candidates_from_meta(request):
    """
    Given job metadata (not stored in DB), find best JobCluster,
    then candidates from correlated CandidateClusters, and return top matches.
    """
    meta_serializer = JobMetaSerializer(data=request.data)
    meta_serializer.is_valid(raise_exception=True)
    job_meta = meta_serializer.validated_data

    top_n = int(request.query_params.get(
        "top_n", request.data.get("top_n", 20)))
    similarity_threshold = float(
        request.query_params.get(
            "similarity_threshold", request.data.get("similarity_threshold", 0.5))
    )

    results = match_candidates_for_job_meta(
        job_meta,
        top_n=top_n,
        similarity_threshold=similarity_threshold,
    )

    return Response(results, status=status.HTTP_200_OK)
