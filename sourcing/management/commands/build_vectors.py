from django.core.management.base import BaseCommand
from sourcing.models import Candidate, Job
from sourcing.services.vectorizer import generate_candidate_vector, generate_job_vector


class Command(BaseCommand):
    help = "Generate vectors for all candidates and jobs"

    def handle(self, *args, **options):
        self.stdout.write("Generating candidate vectors...")
        for cand in Candidate.objects.all():
            generate_candidate_vector(cand)
        self.stdout.write("Generating job vectors...")
        for job in Job.objects.all():
            generate_job_vector(job)
        self.stdout.write(self.style.SUCCESS("Done generating vectors."))
