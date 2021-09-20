#!/usr/bin/env bash

trap /Applications/MAMP/bin/stop.sh SIGKILL SIGINT SIGTERM

URL="http://localhost:8888/content/"
MAMP_DIR="/Applications/MAMP/htdocs/content"

# Copy content to htdocs
echo "Copying..."
mkdir -p $MAMP_DIR
rm -rf $MAMP_DIR/*
cp index.php home.html app-ads.txt $MAMP_DIR

# Start server
echo "Starting server"
/Applications/MAMP/bin/start.sh

open $URL

tail -f /Applications/MAMP/logs/php_error.log
