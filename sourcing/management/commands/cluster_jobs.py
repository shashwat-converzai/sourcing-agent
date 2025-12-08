from django.core.management.base import BaseCommand
from sourcing.services.clustering import cluster_jobs


class Command(BaseCommand):
    help = "Cluster job vectors"

    def add_arguments(self, parser):
        parser.add_argument('--clusters', type=int, default=20)

    def handle(self, *args, **options):
        n_clusters = options['clusters']
        self.stdout.write(f"Clustering jobs into {n_clusters} clusters...")
        cluster_jobs(n_clusters=n_clusters)
        self.stdout.write(self.style.SUCCESS("Done clustering jobs."))
