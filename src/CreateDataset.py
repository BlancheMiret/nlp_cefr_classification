'''
Create a new dataset with new filenames, and removed metadata.
Renamed corpora are in the Datasets/ folder in this repo
Original corpora for all languages can be downloaded from MERLIN website
http://www.merlin-platform.eu/C_data.php
'''

import os

# dirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/" #CZ_ltext_txt", DE_ltext_txt, IT_ltext_txt
# outputdirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/Renamed/"

SCRIPT_DIR =  os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.join(SCRIPT_DIR, "../data/MERLIN Written Learner Corpus for Czech, German, Italian 1.1/merlin-text-v1.1/meta_ltext")
outputdirpath = os.path.join(SCRIPT_DIR, "../data/")

files = os.listdir(dirpath)

# inputdirs = ["CZ_ltext_txt", "DE_ltext_txt", "IT_ltext_txt"]
inputdirs = ["czech", "german", "italian"]
outputdirs = ["CZ","DE","IT"]

for i in range(0, len(inputdirs)):
    files = os.listdir(os.path.join(dirpath,inputdirs[i]))
    for file in files:
        print(file)
        if file.endswith(".txt"):
            content = open(os.path.join(dirpath,inputdirs[i],file),"r").read()
            cefr = content.split("Learner text:")[0].split("Overall CEFR rating: ")[1].split("\n")[0]
            newname = file.replace(".txt","") + "_" + outputdirs[i] + "_" + cefr + ".txt"
            dir = os.path.join(outputdirpath,outputdirs[i])
            if not os.path.exists(dir): ####
                os.makedirs(dir)
            fh = open(os.path.join(outputdirpath,outputdirs[i],newname), "w")
            text = content.split("Learner text:")[1].strip()
            print("wrote: ", os.path.join(outputdirpath,outputdirs[i],newname))
            fh.write(text) ###
            fh.close() ###
