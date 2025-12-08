from django.core.management.base import BaseCommand
from sourcing.services.clustering import cluster_candidates


class Command(BaseCommand):
    help = "Cluster candidate vectors"

    def add_arguments(self, parser):
        parser.add_argument('--clusters', type=int, default=30)

    def handle(self, *args, **options):
        n_clusters = options['clusters']
        self.stdout.write(f"Clustering candidates into {n_clusters} clusters...")
        cluster_candidates(n_clusters=n_clusters)
        self.stdout.write(self.style.SUCCESS("Done clustering candidates."))
