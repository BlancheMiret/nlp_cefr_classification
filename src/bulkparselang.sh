#!/usr/bin/env bash

#Script to parse all files for a given language using its UDPipe model

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

udpipe_file=/../udpipe/src/udpipe
# udpipe_file=$(printf %q "$udpipe_file")

echo $udpipe_file

models_dir="${SCRIPT_DIR}/../Universal Dependencies 2.0 Models for UDPipe (2017-08-01)/udpipe-ud-2.0-170801"
model_de=german-ud-2.0-170801.udpipe
model_cz=czech-ud-2.0-170801.udpipe
model_it=italian-ud-2.0-170801.udpipe

datadir="${SCRIPT_DIR}/../data" #

for lang in CZ IT DE
do

   if [ $lang == "CZ" ]
   then
      modelfile=$model_cz
   elif [ $lang == "IT" ]
   then
      modelfile=$model_it
   elif [ $lang == "DE" ]
   then
      modelfile=$model_de
   fi

   inputdir=${datadir}/${lang}
   outdir=${datadir}/${lang}-Parsed
   mkdir "$outdir"

   files=`ls "$inputdir"`
   for f in $files
   do
       echo $f
       set -x
       ./$udpipe_file --tokenize --tag --parse "${models_dir}/${modelfile}" "${inputdir}/${f}" --output conllu --outfile "${outdir}/${f}.parsed.txt"
       set +x
       echo Wrote ${outdir}/${i}.parsed.txt
       echo
  done
done
