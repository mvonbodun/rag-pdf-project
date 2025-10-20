#!/bin/bash
# save_results.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$TIMESTAMP"

echo "Backing up current results to $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"

cp -r runs/ "$BACKUP_DIR/"
cp -r data/qa/ "$BACKUP_DIR/"

echo "✓ Backup complete: $BACKUP_DIR"
echo "You can now safely re-run the pipeline!"