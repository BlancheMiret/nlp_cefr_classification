#Create a metadata file for MERLIN corpus data, listing all other proficiencylabels.
import os

'''
format:
Rating:
Overall CEFR rating: B1
Grammatical accuracy: B1
Orthography: B2
Vocabulary range: B1
Vocabulary control: B2
Coherence/Cohesion: B1
Sociolinguistic appropriateness: B1
'''

SCRIPT_DIR =  os.path.dirname(os.path.realpath(__file__))
DIRPATH = os.path.join(SCRIPT_DIR, "../data/MERLIN Written Learner Corpus for Czech, German, Italian 1.1/merlin-text-v1.1/meta_ltext")
OUTPATH = os.path.join(SCRIPT_DIR, "../data/")

#Not using this
def list_to_dict(rlist):
    return dict(map(lambda s : s.replace(" ","_").split(':_'), rlist))


#Takes input folder, and stores metadata file for that folder.
def getMetaData(dirpath,inputdir,outpath):
        ratings_dict = {}
        files = os.listdir(os.path.join(dirpath,inputdir))
        fw = open(outpath,"w")
        for file in files:
            #print(file)
            if file.endswith(".txt"):
                content = open(os.path.join(dirpath,inputdir,file),"r").read()
                ratings = content.split("Rating:")[1].split("\n\n")[0].strip().split("\n") #get all ratings in a list
                #join the list contents to a string, and remove white spaces and slashes in descriptions.
                fw.write(file.replace(".txt","")+","+",".join(ratings).replace(" ","").replace("/",""))
                fw.write("\n")

        fw.close()


def main():
    # dirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/"
    dirpath = DIRPATH
    #inputdirs = ["CZ_ltext_txt", "DE_ltext_txt", "IT_ltext_txt"]
    inputdirs = ["czech", "german", "italian"]
    outputnames = ["CZMetadata.txt","DEMetadata.txt","ITMetadata.txt"]
    for i in range(0, len(inputdirs)):
        # outpath = os.path.join(dirpath,outputnames[i])
        outpath = os.path.join(OUTPATH,outputnames[i])
        getMetaData(dirpath,inputdirs[i],outpath)
        print("Wrote to: ", outpath)

main()
