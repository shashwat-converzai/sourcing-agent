python manage.py shell -c "
from django.core.management import call_command;
call_command('seed_dummy_data');
call_command('build_vectors');
call_command('cluster_candidates');
call_command('cluster_jobs');
call_command('compute_correlations');
print('✨ FULL PIPELINE COMPLETE ✨');
"
