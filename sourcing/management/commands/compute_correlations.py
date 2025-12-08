from django.core.management.base import BaseCommand
from sourcing.services.correlation import compute_cluster_correlations


class Command(BaseCommand):
    help = "Compute correlation between job and candidate clusters"

    def handle(self, *args, **options):
        self.stdout.write("Computing cluster correlations...")
        compute_cluster_correlations()
        self.stdout.write(self.style.SUCCESS("Done computing correlations."))
