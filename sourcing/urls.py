from django.urls import path
from . import views

urlpatterns = [
    path('jobs/', views.list_jobs, name='job-list'),
    path('jobs/<str:job_id>/', views.job_detail, name='job-detail'),
    path('jobs/<str:job_id>/auto-source/',
         views.auto_source_job, name='job-auto-source'),
    path('match-candidates/', views.match_candidates_from_meta,
         name='match-candidates-meta'),
]
