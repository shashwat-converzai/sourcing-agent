from django.db import models


class Skill(models.Model):
    name = models.CharField(max_length=128, unique=True)

    def __str__(self):
        return self.name


class Candidate(models.Model):
    candidate_id = models.CharField(max_length=64, unique=True)
    full_name = models.CharField(max_length=255)
    email = models.EmailField(null=True, blank=True)
    phone = models.CharField(max_length=64, null=True, blank=True)

    resume_raw = models.TextField(null=True, blank=True)

    present_location = models.CharField(max_length=255, null=True, blank=True)
    permanent_location = models.CharField(max_length=255, null=True, blank=True)

    total_experience_years = models.FloatField(null=True, blank=True)
    current_designation = models.CharField(max_length=255, null=True, blank=True)
    current_company = models.CharField(max_length=255, null=True, blank=True)

    notice_period_days = models.IntegerField(null=True, blank=True)
    salary_current = models.IntegerField(null=True, blank=True)
    salary_expected = models.IntegerField(null=True, blank=True)

    # taxonomy fields
    industry = models.CharField(max_length=64, null=True, blank=True)        # IT / Logistics / Healthcare etc.
    domain = models.CharField(max_length=64, null=True, blank=True)          # software / hr / finance etc.
    seniority_level = models.CharField(max_length=32, null=True, blank=True) # junior / mid-level / mid-senior / senior

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.full_name} ({self.candidate_id})"


class CandidateSkill(models.Model):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"

    PROFICIENCY_CHOICES = [
        (BEGINNER, "Beginner"),
        (INTERMEDIATE, "Intermediate"),
        (EXPERT, "Expert"),
    ]

    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="candidate_skills")
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE)
    proficiency = models.CharField(max_length=32, choices=PROFICIENCY_CHOICES, null=True, blank=True)
    years_of_experience = models.FloatField(null=True, blank=True)
    last_used_year = models.IntegerField(null=True, blank=True)

    class Meta:
        unique_together = ("candidate", "skill")

    def __str__(self):
        return f"{self.candidate.full_name} - {self.skill.name}"


class Experience(models.Model):
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="experiences")
    company_name = models.CharField(max_length=255)
    designation = models.CharField(max_length=255)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    responsibilities = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.candidate.full_name} @ {self.company_name}"


class Certification(models.Model):
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="certifications")
    name = models.CharField(max_length=255)
    issuing_org = models.CharField(max_length=255, null=True, blank=True)
    issued_date = models.DateField(null=True, blank=True)
    expiry_date = models.DateField(null=True, blank=True)
    credential_id = models.CharField(max_length=255, null=True, blank=True)
    credential_url = models.URLField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} - {self.candidate.full_name}"


class Job(models.Model):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"

    LOCATION_TYPE_CHOICES = [
        (REMOTE, "Remote"),
        (HYBRID, "Hybrid"),
        (ONSITE, "OnSite"),
    ]

    OPEN = "open"
    ON_HOLD = "on_hold"
    CLOSED = "closed"

    STATUS_CHOICES = [
        (OPEN, "Open"),
        (ON_HOLD, "On hold"),
        (CLOSED, "Closed"),
    ]

    job_id = models.CharField(max_length=64, unique=True)
    title = models.CharField(max_length=255)
    role_code = models.CharField(max_length=128, null=True, blank=True)

    description = models.TextField(null=True, blank=True)

    location = models.CharField(max_length=255, null=True, blank=True)
    location_type = models.CharField(
        max_length=32, choices=LOCATION_TYPE_CHOICES, default=ONSITE
    )

    experience_min_years = models.FloatField(null=True, blank=True)
    experience_max_years = models.FloatField(null=True, blank=True)

    budget_min = models.IntegerField(null=True, blank=True)
    budget_max = models.IntegerField(null=True, blank=True)

    industry = models.CharField(max_length=64, null=True, blank=True)
    domain = models.CharField(max_length=64, null=True, blank=True)
    seniority_level = models.CharField(max_length=32, null=True, blank=True)

    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=OPEN)
    priority = models.CharField(max_length=32, default="medium")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.job_id})"


class JobSkill(models.Model):
    MUST_HAVE = "must"
    NICE_TO_HAVE = "nice"

    IMPORTANCE_CHOICES = [
        (MUST_HAVE, "Must have"),
        (NICE_TO_HAVE, "Nice to have"),
    ]

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="job_skills")
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE)
    importance = models.CharField(max_length=16, choices=IMPORTANCE_CHOICES, default=MUST_HAVE)

    class Meta:
        unique_together = ("job", "skill")

    def __str__(self):
        return f"{self.job.title} - {self.skill.name} ({self.importance})"


class CandidateJobApplication(models.Model):
    APPLIED = "applied"
    INTERESTED = "interested"
    RECOMMENDED = "recommended"
    SUBMITTED = "submitted"
    PLACED = "placed"
    REJECTED = "rejected"
    NOT_INTERESTED = "not_interested"

    STATUS_CHOICES = [
        (APPLIED, "Applied"),
        (INTERESTED, "Interested"),
        (RECOMMENDED, "Recommended"),
        (SUBMITTED, "Submitted"),
        (PLACED, "Placed"),
        (REJECTED, "Rejected"),
        (NOT_INTERESTED, "Not Interested"),
    ]

    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="applications")
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="applications")
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=INTERESTED)
    source = models.CharField(max_length=128, null=True, blank=True)
    notes = models.TextField(null=True, blank=True)

    last_contacted_at = models.DateTimeField(null=True, blank=True)
    last_status_change_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("candidate", "job")

    def __str__(self):
        return f"{self.candidate.full_name} -> {self.job.title} ({self.status})"


class CandidateVector(models.Model):
    candidate = models.OneToOneField(
        Candidate, on_delete=models.CASCADE, related_name="vector"
    )
    vector = models.JSONField()                     # list of floats
    semantic_vector = models.JSONField(null=True, blank=True)
    structured_vector = models.JSONField(null=True, blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"CandidateVector({self.candidate_id})"


class JobVector(models.Model):
    job = models.OneToOneField(
        Job, on_delete=models.CASCADE, related_name="vector"
    )
    vector = models.JSONField()
    semantic_vector = models.JSONField(null=True, blank=True)
    structured_vector = models.JSONField(null=True, blank=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"JobVector({self.job_id})"


class CandidateCluster(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    center_vector = models.JSONField()
    size = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CandidateCluster({self.id}, size={self.size})"


class JobCluster(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    center_vector = models.JSONField()
    size = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"JobCluster({self.id}, size={self.size})"


class CandidateClusterMembership(models.Model):
    candidate = models.OneToOneField(
        Candidate, on_delete=models.CASCADE, related_name="cluster_membership"
    )
    cluster = models.ForeignKey(
        CandidateCluster, on_delete=models.CASCADE, related_name="members"
    )
    distance_to_center = models.FloatField()

    def __str__(self):
        return f"{self.candidate_id} -> cluster {self.cluster_id}"


class JobClusterMembership(models.Model):
    job = models.OneToOneField(
        Job, on_delete=models.CASCADE, related_name="cluster_membership"
    )
    cluster = models.ForeignKey(
        JobCluster, on_delete=models.CASCADE, related_name="members"
    )
    distance_to_center = models.FloatField()

    def __str__(self):
        return f"{self.job_id} -> cluster {self.cluster_id}"


class ClusterCorrelation(models.Model):
    job_cluster = models.ForeignKey(
        JobCluster, on_delete=models.CASCADE, related_name="candidate_correlations"
    )
    candidate_cluster = models.ForeignKey(
        CandidateCluster, on_delete=models.CASCADE, related_name="job_correlations"
    )
    correlation = models.FloatField()

    class Meta:
        unique_together = ("job_cluster", "candidate_cluster")
        indexes = [
            models.Index(fields=["job_cluster", "-correlation"]),
        ]

    def __str__(self):
        return f"Corr(J{self.job_cluster_id}, C{self.candidate_cluster_id})={self.correlation:.3f}"


class JobCandidateMatch(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="matches")
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name="matches")

    score = models.FloatField()
    skill_match_score = models.FloatField(null=True, blank=True)
    semantic_match_score = models.FloatField(null=True, blank=True)
    experience_match_score = models.FloatField(null=True, blank=True)
    location_match_score = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("job", "candidate")
        indexes = [
            models.Index(fields=["job", "-score"]),
        ]

    def __str__(self):
        return f"{self.job.title} <-> {self.candidate.full_name} ({self.score:.3f})"
