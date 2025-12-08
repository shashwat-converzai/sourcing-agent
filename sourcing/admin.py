from django.contrib import admin
from . import models


class SkillAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Skill._meta.fields]
    readonly_fields = [field.name for field in models.Skill._meta.fields]

    def get_meta_info(self, obj):
        return str(models.Skill._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Candidate._meta.fields]
    readonly_fields = [field.name for field in models.Candidate._meta.fields]

    def get_meta_info(self, obj):
        return str(models.Candidate._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateSkillAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.CandidateSkill._meta.fields]
    readonly_fields = [
        field.name for field in models.CandidateSkill._meta.fields]

    def get_meta_info(self, obj):
        return str(models.CandidateSkill._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class ExperienceAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Experience._meta.fields]
    readonly_fields = [field.name for field in models.Experience._meta.fields]

    def get_meta_info(self, obj):
        return str(models.Experience._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CertificationAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Certification._meta.fields]
    readonly_fields = [
        field.name for field in models.Certification._meta.fields]

    def get_meta_info(self, obj):
        return str(models.Certification._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Job._meta.fields]
    readonly_fields = [field.name for field in models.Job._meta.fields]

    def get_meta_info(self, obj):
        return str(models.Job._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobSkillAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.JobSkill._meta.fields]
    readonly_fields = [field.name for field in models.JobSkill._meta.fields]

    def get_meta_info(self, obj):
        return str(models.JobSkill._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateJobApplicationAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.CandidateJobApplication._meta.fields]
    readonly_fields = [
        field.name for field in models.CandidateJobApplication._meta.fields]

    def get_meta_info(self, obj):
        return str(models.CandidateJobApplication._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateVectorAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.CandidateVector._meta.fields]
    readonly_fields = [
        field.name for field in models.CandidateVector._meta.fields]

    def get_meta_info(self, obj):
        return str(models.CandidateVector._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobVectorAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.JobVector._meta.fields]
    readonly_fields = [field.name for field in models.JobVector._meta.fields]

    def get_meta_info(self, obj):
        return str(models.JobVector._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateClusterAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.CandidateCluster._meta.fields]
    readonly_fields = [
        field.name for field in models.CandidateCluster._meta.fields]

    def get_meta_info(self, obj):
        return str(models.CandidateCluster._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobClusterAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.JobCluster._meta.fields]
    readonly_fields = [field.name for field in models.JobCluster._meta.fields]

    def get_meta_info(self, obj):
        return str(models.JobCluster._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class CandidateClusterMembershipAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.CandidateClusterMembership._meta.fields]
    readonly_fields = [
        field.name for field in models.CandidateClusterMembership._meta.fields]

    def get_meta_info(self, obj):
        return str(models.CandidateClusterMembership._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobClusterMembershipAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.JobClusterMembership._meta.fields]
    readonly_fields = [
        field.name for field in models.JobClusterMembership._meta.fields]

    def get_meta_info(self, obj):
        return str(models.JobClusterMembership._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class ClusterCorrelationAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.ClusterCorrelation._meta.fields]
    readonly_fields = [
        field.name for field in models.ClusterCorrelation._meta.fields]

    def get_meta_info(self, obj):
        return str(models.ClusterCorrelation._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


class JobCandidateMatchAdmin(admin.ModelAdmin):
    list_display = [
        field.name for field in models.JobCandidateMatch._meta.fields]
    readonly_fields = [
        field.name for field in models.JobCandidateMatch._meta.fields]

    def get_meta_info(self, obj):
        return str(models.JobCandidateMatch._meta.__dict__)
    get_meta_info.short_description = "Meta Info"
    fields = list_display + ["get_meta_info"]


admin.site.register(models.Skill, SkillAdmin)
admin.site.register(models.Candidate, CandidateAdmin)
admin.site.register(models.Job, JobAdmin)
admin.site.register(models.CandidateSkill, CandidateSkillAdmin)
admin.site.register(models.JobSkill, JobSkillAdmin)
admin.site.register(models.CandidateJobApplication,
                    CandidateJobApplicationAdmin)
admin.site.register(models.CandidateVector, CandidateVectorAdmin)
admin.site.register(models.JobVector, JobVectorAdmin)
admin.site.register(models.CandidateCluster, CandidateClusterAdmin)
admin.site.register(models.JobCluster, JobClusterAdmin)
admin.site.register(models.CandidateClusterMembership,
                    CandidateClusterMembershipAdmin)
admin.site.register(models.JobClusterMembership, JobClusterMembershipAdmin)
admin.site.register(models.ClusterCorrelation, ClusterCorrelationAdmin)
admin.site.register(models.JobCandidateMatch, JobCandidateMatchAdmin)
