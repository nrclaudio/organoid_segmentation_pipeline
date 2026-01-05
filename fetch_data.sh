#!/bin/bash

# Configuration
SERVER="work-via-jump"
REMOTE_PATH="/exports/nieromics-hpc/cnovellarausell/organoid_tx_realigned_files"

echo "Connecting to $SERVER to retrieve directory list..."

# 1. Get the list of 'realigned_' directories
DIRS=$(ssh "$SERVER" "ls -1 $REMOTE_PATH | grep 'realigned_'")

if [ -z "$DIRS" ]; then
    echo "Error: No directories found. Check your connection or path."
    exit 1
fi

# 2. Loop through each directory found
echo "$DIRS" | while read PARENT_DIR; do
    
    echo "------------------------------------------------"
    echo "Processing: $PARENT_DIR"

    # CRITICAL FIX: Added '-n' to ssh so it doesn't break the loop
    SUBFOLDER=$(ssh -n "$SERVER" "ls -1 $REMOTE_PATH/$PARENT_DIR" | head -n 1)

    if [ -z "$SUBFOLDER" ]; then
        echo "  ! Warning: No subfolder found in $PARENT_DIR. Skipping."
        continue
    fi

    echo "  > Detected subfolder: $SUBFOLDER"

    mkdir -p "./$PARENT_DIR/outs"

    echo "  > Syncing 'feature_expression' and 'image'..."
    
    # We also add -n here just to be safe, though rsync usually handles this better
    rsync -azm \
        --include='feature_expression/***' \
        --include='image/***' \
        --exclude='*' \
        "$SERVER:$REMOTE_PATH/$PARENT_DIR/$SUBFOLDER/outs/" \
        "./$PARENT_DIR/outs/"

done

echo "------------------------------------------------"
echo "All downloads complete."
