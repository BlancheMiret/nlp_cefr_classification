"""
Purpose: Knowing error statistics in the data for DE and IT using LanguageTool
"""

# import language_check
import language_tool_python
import os, collections, pprint

def write_featurelist(file_path,some_list):
    fh = open(file_path, "w")
    for item in some_list:
      fh.write(item)
      fh.write("\n")
    fh.close()

def error_stats(inputpath,lang,output_path):
    files = os.listdir(inputpath)
    # checker = language_check.LanguageTool(lang)
    checker = language_tool_python.LanguageTool(lang)
    rules = {}
    locqualityissuetypes = {}
    categories = {}

    for file in files:
        if file.endswith(".txt"):
            text = open(os.path.join(inputpath,file)).read()
            matches = checker.check(text)
            for match in matches:
                print(match)
                print(match.__dict__)
                rule = match.ruleId
                #loc = match.locqualityissuetype
                cat = match.category
                rules[rule] = rules.get(rule,0) +1
                #locqualityissuetypes[loc] = locqualityissuetypes.get(loc,0) +1
                categories[cat] = categories.get(cat,0)+1

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    write_featurelist(output_path+lang+"-rules.txt", sorted(rules.keys()))
    #write_featurelist(output_path+lang+"-locquality.txt", sorted(locqualityissuetypes.keys()))
    write_featurelist(output_path+lang+"-errorcats.txt", sorted(categories.keys()))

# inputpath_de = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/DE/"
# inputpath_it = "/home/bangaru/GitProjects/CrossLingualScoring/Datasets/IT/"

SCRIPT_DIR =  os.path.dirname(os.path.realpath(__file__))
inputpath_de = os.path.join(SCRIPT_DIR, "../data/DE/")
inputpath_it = os.path.join(SCRIPT_DIR, "../data/IT/")
outpath = os.path.join(SCRIPT_DIR, "../data/features/")

error_stats(inputpath_de, "de", outpath)
error_stats(inputpath_it, "it", outpath)
