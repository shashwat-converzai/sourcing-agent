from rest_framework import serializers
from .models import Job, Candidate, CandidateJobApplication


class CandidateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candidate
        fields = [
            "id", "candidate_id", "full_name", "email", "phone",
            "present_location", "permanent_location",
            "total_experience_years", "current_designation",
            "industry", "domain", "seniority_level",
        ]


class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = [
            "id", "job_id", "title", "role_code",
            "location", "location_type",
            "experience_min_years", "experience_max_years",
            "industry", "domain", "seniority_level",
            "status", "priority",
        ]


class CandidateJobApplicationSerializer(serializers.ModelSerializer):
    candidate = CandidateSerializer()
    job = JobSerializer()

    class Meta:
        model = CandidateJobApplication
        fields = ["id", "candidate", "job", "status", "source", "notes"]


class JobMetaSerializer(serializers.Serializer):
    """
    Input payload for ad-hoc job metadata (no DB Job created).
    """
    title = serializers.CharField()
    description = serializers.CharField(allow_blank=True, required=False)
    location = serializers.CharField(allow_blank=True, required=False)

    industry = serializers.CharField(allow_blank=True, required=False)
    domain = serializers.CharField(allow_blank=True, required=False)
    seniority_level = serializers.CharField(allow_blank=True, required=False)

    experience_min_years = serializers.FloatField(required=False)
    experience_max_years = serializers.FloatField(required=False)

    required_skills = serializers.ListField(
        child=serializers.CharField(), required=False
    )
