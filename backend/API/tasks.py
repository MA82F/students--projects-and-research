from celery import shared_task
import os
import time
import glob
from django.conf import settings

@shared_task
def cleanup_old_temp_files(max_age_hours=1):
    """
    Background task to clean up old temporary result files
    """
    max_age_seconds = max_age_hours * 3600
    temp_dir = os.path.join(settings.BASE_DIR, 'temp_results')
    
    if not os.path.exists(temp_dir):
        return f"Temp directory does not exist: {temp_dir}"

    current_time = time.time()
    pattern = os.path.join(temp_dir, '*.json')
    deleted_count = 0

    for file_path in glob.glob(pattern):
        try:
            file_age = current_time - os.path.getmtime(file_path)
            
            if file_age > max_age_seconds:
                os.remove(file_path)
                deleted_count += 1
                
        except OSError:
            continue  # Skip files that can't be deleted

    return f"Cleanup completed. Deleted {deleted_count} old files."

@shared_task
def cleanup_single_file(file_path, delay_seconds=300):
    """
    Background task to clean up a specific file after a delay
    This is a fallback in case the main cleanup fails
    """
    import time
    time.sleep(delay_seconds)  # Wait 5 minutes
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"Delayed cleanup: Removed {file_path}"
    except OSError:
        return f"Failed to remove {file_path}"
    
    return f"File {file_path} was already removed"
