#!/bin/bash

files='PATH_TO_3RScan_FOLDER/3RScan/*/*.obj'


for file in $files
do
  echo $file
  cloudcompare.CloudCompare -SILENT -O $file -SAMPLE_MESH DENSITY 10000 -CLEAR_MESHES -C_EXPORT_FMT ASC -ADD_HEADER -ADD_PTS_COUNT -AUTO_SAVE ON -SAVE_CLOUDS
done
