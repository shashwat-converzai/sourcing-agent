from django.core.management.base import BaseCommand
from django.shortcuts import get_object_or_404

from sourcing.models import Job
from sourcing.services.agent import auto_source_for_job


class Command(BaseCommand):
    help = "Auto-source candidates for a given job_id"

    def add_arguments(self, parser):
        parser.add_argument('job_id', type=str)
        parser.add_argument('--top_n', type=int, default=20)
        parser.add_argument('--threshold', type=float, default=0.5)

    def handle(self, *args, **options):
        job_id = options['job_id']
        top_n = options['top_n']
        threshold = options['threshold']

        job = get_object_or_404(Job, job_id=job_id)
        self.stdout.write(f"Auto-sourcing candidates for job {job.job_id} ({job.title})...")
        results = auto_source_for_job(job, top_n=top_n, similarity_threshold=threshold)
        self.stdout.write(self.style.SUCCESS(f"Found {len(results)} candidates."))
        for r in results:
            self.stdout.write(f"- {r['full_name']} ({r['candidate_id']}), sim={r['similarity']:.3f}")
