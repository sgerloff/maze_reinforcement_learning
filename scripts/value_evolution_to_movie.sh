#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input directory> <output file mp4/gif> <max number of frames>"
else
  ffmpeg -i $1/value_map_t_%d.png -frames:v $3 $2
fi