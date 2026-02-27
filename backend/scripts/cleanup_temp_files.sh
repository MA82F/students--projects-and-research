#!/bin/bash
# Cron job script to clean up old temporary files
# Add this to crontab: */30 * * * * /path/to/cleanup_temp_files.sh

cd /path/to/your/project/backend
source venv/bin/activate
python manage.py cleanup_temp_files --max-age=3600

# Alternative: Clean files older than 1 hour directly
# find /path/to/your/project/backend/temp_results -name "*.json" -mtime +1 -delete
