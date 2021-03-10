#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input directory> <output file mp4/gif>"
else
  ffmpeg -i $1/run_t_%d.png $2
fi