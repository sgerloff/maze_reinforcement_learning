#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input directory> <output file name>"
else
  ffmpeg -i $1/run_t_%d.png $2.mp4
  ffmpeg -i $2.mp4 -vf "fps=10,scale=700:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 $2.gif
  rm $2.mp4
fi