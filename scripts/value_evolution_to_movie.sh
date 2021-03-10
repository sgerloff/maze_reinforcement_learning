#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input directory> <output file name> <max number of frames>"
else
  ffmpeg -i $1/value_map_t_%d.png -frames:v $3 $2
  ffmpeg -i $2.mp4 -vf "fps=10,scale=700:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $2.gif
  rm $2.mp4
fi