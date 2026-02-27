import os
import time
import glob
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Clean up old temporary result files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--max-age',
            type=int,
            default=3600,  # 1 hour in seconds
            help='Maximum age of files to keep (in seconds)'
        )

    def handle(self, *args, **options):
        max_age = options['max_age']
        temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
        
        if not os.path.exists(temp_dir):
            self.stdout.write('Temp directory does not exist')
            return

        current_time = time.time()
        pattern = os.path.join(temp_dir, '*.json')
        deleted_count = 0

        for file_path in glob.glob(pattern):
            try:
                # Check file age
                file_age = current_time - os.path.getmtime(file_path)
                
                if file_age > max_age:
                    os.remove(file_path)
                    deleted_count += 1
                    self.stdout.write(f'Deleted: {os.path.basename(file_path)}')
                    
            except OSError as e:
                self.stdout.write(f'Error deleting {file_path}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Cleanup completed. Deleted {deleted_count} old files.')
        )
