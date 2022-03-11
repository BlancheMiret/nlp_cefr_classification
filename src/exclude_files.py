import os
import glob

SCRIPT_DIR =  os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR,"../data/")
REMOVED_FILES_DIR = os.path.join(SCRIPT_DIR, "../data/RemovedFiles/")

TO_EXCLUDE = [
    "0909_CZ_EMPTY.txt",
    "1385_0001743_IT_unrated.txt",
    "PHA0209025_CZ_C1.txt",
    "0909_CZ_EMPTY.txt.parsed.txt",
    "1385_0001743_IT_unrated.txt.parsed.txt",
    "PHA0209025_CZ_C1.txt.parsed.txt",
    "1031_0002007_DE_C2.txt",
    "1385_0001745_IT_unrated.txt",
    "PHA0411057_CZ_C1.txt",
    "1031_0002007_DE_C2.txt.parsed.txt",
    "1385_0001745_IT_unrated.txt.parsed.txt",
    "PHA0411057_CZ_C1.txt.parsed.txt",
    "1031_0003028_DE_C2.txt",
    "1385_0001769_IT_unrated.txt",
    "PHA0710020_CZ_EMPTY.txt",
    "1031_0003028_DE_C2.txt.parsed.txt",
    "1385_0001769_IT_unrated.txt.parsed.txt",
    "PHA0710020_CZ_EMPTY.txt.parsed.txt",
    "1031_0003044_DE_C2.txt",
    "1395_0000590_IT_unrated.txt",
    "PHA0811018_CZ_C1.txt",
    "1031_0003044_DE_C2.txt.parsed.txt",
    "1395_0000590_IT_unrated.txt.parsed.txt",
    "PHA0811018_CZ_C1.txt.parsed.txt",
    "1031_0003045_DE_C2.txt",
    "1395_0000605_IT_unrated.txt",
    "PHA1109010_CZ_A1.txt",
    "1031_0003045_DE_C2.txt.parsed.txt",
    "1395_0000605_IT_unrated.txt.parsed.txt",
    "PHA1109010_CZ_A1.txt.parsed.txt",
    "1325_9000135_IT_B2.txt",
    "1395_0000638_IT_unrated.txt",
    "PHA1110018_CZ_C1.txt",
    "1325_9000135_IT_B2.txt.parsed.txt",
    "1395_0000638_IT_unrated.txt.parsed.txt",
    "PHA1110018_CZ_C1.txt.parsed.txt",
    "1325_9000613_IT_B2.txt",
    "1395_0001107_IT_EMPTY.txt",
    "1325_9000613_IT_B2.txt.parsed.txt",
    "1395_0001107_IT_EMPTY.txt.parsed.txt"
]

def main():

    if not os.path.exists(REMOVED_FILES_DIR):
        os.makedirs(REMOVED_FILES_DIR)

    for path in glob.glob(os.path.join(DATA_DIR, "*/*.txt")):
        if any([f in path for f in TO_EXCLUDE]):
            bs = os.path.basename(path)
            os.rename(path, os.path.join(REMOVED_FILES_DIR, bs))
            print(f"Moved {bs} to removed files.")

if __name__ == '__main__':
    main()
