# NLP CEFR Classification



`sqljdf` 

## Installation

- Create a virtual environment and activate it:

```
$ virtualenv -p python3 .env
$ source .env/bin/activate
```

- Install the requirements:

```
$ pip3 install -r requirements.txt
```

## Usage

**Steps**

1. Download data from [MERLIN website](https://clarin.eurac.edu/repository/xmlui/handle/20.500.12124/6) in the `data` repertory. Name of the downloaded (and decompressed) repertory should be `MERLIN Written Learner Corpus for Czech, German, Italian 1`.

2. Create structured dataset :

   - `$ python3 src/CreateDataset.py`
- Use input data from `meta_ltext` rep in the original data dir. Extract only the text. Create reps `CZ`, `IT`and `DE` in `data`rep with all input files named with original id name + code lang + AESR general level. Ex : `1023_0001416_DE_B2.txt`
   - Notes:

     -  script missed a few lines (did not write the output text)

     - 441 files in CZ vs 434 in the delivered code (miss 7 -> in `RemovedFiles`)

     - 1033 vs 1029 in DE (miss 4 -> in `RemovedFiles`)

     - 813 vs 804 in IT (miss 9 -> in `RemovedFiles`)

3. Extract metadata:

   - `$ python3 src/CreateMetadataFile.py`
   - Extract metadata from same files as 2.. Create 1 metadata file per language in `data` rep. One file line example : `0601,OverallCEFRrating:A2,Grammaticalaccuracy:A2,Orthography:B1,Vocabularyrange:B1,Vocabularycontrol:A2,CoherenceCohesion:B1,Sociolinguisticappropriateness:A2`
   - Notes: 
     - OK

4. Extract language features with `language-tool`

   - `$ python3 ErrorStats.py`

   - Extract features for Czech and Deutsch with package `language_check`. Creates in `features` rep 3 files by lang : 
     - `error cats.txt` : catégories d'erreurs trouvées dans les textes **Only the files ending with errorcats.txt have been used in this paper.** Donc de toute façon pas grave pour le package.
     - `locquality.txt`, type d'erreur (?)
     - `rule.txt` : règle correspondant aux erreurs
   - Notes:
     - install `language_check`: le package ne semble plus utilisé. erreur :
       - --> besoin de JDK 6
       - ou utiliser le package `language_tool_python` à la place
   
   
   ```
   (.env_nlp) blanchemiret@Blanche src (main) $ pip3 install language-check
   Collecting language-check
     Using cached language-check-1.1.tar.gz (33 kB)
     Preparing metadata (setup.py) ... done
   Building wheels for collected packages: language-check
     Building wheel for language-check (setup.py) ... error
     error: subprocess-exited-with-error
     
     × python setup.py bdist_wheel did not run successfully.
     │ exit code: 1
     ╰─> [4 lines of output]
         Could not parse Java version from """openjdk version "15.0.1" 2020-10-20
         OpenJDK Runtime Environment (build 15.0.1+9-18)
         OpenJDK 64-Bit Server VM (build 15.0.1+9-18, mixed mode, sharing)
         """.
         [end of output]
     
     note: This error originates from a subprocess, and is likely not a problem with pip.
     ERROR: Failed building wheel for language-check
     Running setup.py clean for language-check
   Failed to build language-check
   Installing collected packages: language-check
     Running setup.py install for language-check ... error
     error: subprocess-exited-with-error
     
     × Running setup.py install for language-check did not run successfully.
     │ exit code: 1
     ╰─> [4 lines of output]
         Could not parse Java version from """openjdk version "15.0.1" 2020-10-20
         OpenJDK Runtime Environment (build 15.0.1+9-18)
         OpenJDK 64-Bit Server VM (build 15.0.1+9-18, mixed mode, sharing)
         """.
         [end of output]
     
     note: This error originates from a subprocess, and is likely not a problem with pip.
   error: legacy-install-failure
   
   × Encountered error while trying to install package.
   ╰─> language-check
   
   note: This is an issue with the package mentioned above, not pip.
   hint: See above for output from the failure.
   ```
   
   - essai avec `language-tool-python`(suivant [cette recommandation](https://stackoverflow.com/questions/67460277/error-installing-language-check-using-pip-install))
   
   - avec le nouveau package, l'attribut `locqualityissuetype`n'existe pas 
     - --> utilisation de l'attribut `ruleIssueType` à la place qui semble correspondre.

5. Parse data with UDPIPE

   - Install `udpipe`:  Install in commande line : 
     - Installation manual page : https://ufal.mff.cuni.cz/udpipe/1/install
     - Clone `dep` https://github.com/ufal/udpipe **in the root dir of this rep** and run `make` in the `src`directory --> a `udpipe` binary is created in `./udpipe/udpipe`

   - Download Czech, italian and Deutsch trained models : 
     - Model manual page : https://ufal.mff.cuni.cz/udpipe/1/models 
     - Download all 2.0 models here : https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2364?show=full : click on "Download all files in items", then unzip the downloaded archive **in the root dir of this rep**. , then `udpipe-ud-2.0-170801`. Models should now be available in `./Universal Dependencies 2.0 Models for UDPipe (2017-08-01)/udpipe-ud-2.0-170801/...`
   - `./bulk...`
   - Notes
     - Pas si facile de comprendre comment installer, ni de télécharger les modèles
     - Les versions des modèles UDPIPE ne sont pas clairement précisées, vu en commentaire d'un script la commande utilisée, le chemin du modèle indiquait la version 2.0 pour le modèle tchèque --> utilisation modèle 2.0 pour les 3 modèles 

6. **MONOLINGUAL_CV.PY** --> correspond à "Embeddings" (yes à priori)

   - Use plain text as input 

   - Notes : 

     - `keras`:

       - `pip3 install keras`

         - ```
           Traceback (most recent call last):
             File "monolingual_cv.py", line 5, in <module>
               from keras.preprocessing import sequence
             File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/keras/__init__.py", line 25, in <module>
               from keras import models
             File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/keras/models.py", line 19, in <module>
               from keras import backend
             File "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/keras/backend.py", line 39, in <module>
               from tensorflow.python.eager.context import get_config
           ImportError: cannot import name 'get_config' from 'tensorflow.python.eager.context' (/Users/blanchemiret/Library/Python/3.7/lib/python/site-packages/tensorflow/python/eager/context.py)
           ```

         - --> pip install --upgrade tensorflow

         - :ok: 

     - Attention : `model.predict_classes`, function was removed in tensorflow 2.6. had to replace the function in the code. No package version specified in code.

   - Usage : `python3 monolingual_cv.py ../data/DE/`

     - 10-fold cross validation
     - MLP with only Embedding and Dense softmax output function : classification task with MLP

7. **MULTI_LINGUAL.PY** : donc semble prédire aussi le code langue :  multi-lingual, multi-task learning (learning the language, and learning its CEFR). Avec des embeddings de mot ET caractères (concatène)

   - Notes : :ok: à l'exécution , mais score correspond pas

8. multi_lingual_no_langfeat.py

   - Notes: :ok:  à l'exécution, mais score correspond pas
   
8. `IDEA_POC.PY`

   - Notes:
   
     - ```
       (.env_nlp) blanchemiret@Blanche src (bmi_working_branch) $ python3 IdeaPOC.py 
       Traceback (most recent call last):
         File "IdeaPOC.py", line 8, in <module>
           from sklearn.preprocessing import Imputer #to replace NaN with mean values.
       ImportError: cannot import name 'Imputer' from 'sklearn.preprocessing' (/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/preprocessing/__init__.py)
       ```
   
       - --> https://stackoverflow.com/questions/59439096/importerror-cannnot-import-name-imputer-from-sklearn-preprocessing, mais ne semble pas utilisé dans le code, toutes les lignes y faisant appel sont commentées
   
     - **L'extraction des domain features est problématique**
   
       - Pas de souci pour les expériences multilingues : pas d'extraction des domain features dans ce mode. En revanche, pour monolingue et crosslingue, on essaie d'extraire les fameuses "domain features" pour l'allemand et le tchèque. L'exécution semblait prendre un temps bien trop long. J'ai ajouté des messages pour comprendre ce qui prenait du temps :
   
       ```
       ##### MONOLINGUAL MODELS - FEATURES: all - LANG : DE #####
       In getScoringFeatures
       1091_0000061_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 12.56604790687561 sec.
       1061_0120426_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 13.215233564376831 sec.
       1023_0109606_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 14.215504884719849 sec.
       1091_0000055_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 15.186673879623413 sec.
       1071_0248336_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 16.56483793258667 sec.
       1071_0024763_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 18.431652307510376 sec.
       1023_0108813_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 20.70341420173645 sec.
       1071_0024757_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 21.737897157669067 sec.
       1061_0120874_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 24.081716060638428 sec.
       1071_0024685_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 25.613893032073975 sec.
       1031_0003336_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 25.636111974716187 sec.
       1031_0003141_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 27.713420152664185 sec.
       1031_0003221_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 30.019333124160767 sec.
       1061_0120359_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 32.025243043899536 sec.
       1061_0120314_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 31.764833211898804 sec.
       1091_0000140_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 31.3243248462677 sec.
       1091_0000020_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 33.44585204124451 sec.
       1061_0120494_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 34.87455105781555 sec.
       1071_0024826_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 37.39989924430847 sec.
       1091_0000263_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 38.28219485282898 sec.
       1023_0101855_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 39.803815841674805 sec.
       1071_0242043_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 40.87883377075195 sec.
       1071_0024689_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 41.463289737701416 sec.
       1061_0120878_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 44.38386392593384 sec.
       1023_0109891_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 46.22001004219055 sec.
       1031_0003179_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 48.443629026412964 sec.
       1091_0000016_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 46.46049094200134 sec.
       1023_0103836_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 49.38459920883179 sec.
       1091_0000101_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 48.86769700050354 sec.
       1091_0000022_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 50.10604381561279 sec.
       1071_0024854_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 54.113966941833496 sec.
       1091_0000145_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 59.44367694854736 sec.
       1071_0248305_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 56.91392731666565 sec.
       1061_0120368_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 56.52687692642212 sec.
       1091_0000052_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 58.56247115135193 sec.
       1091_0000171_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 60.20961785316467 sec.
       1023_0109026_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 61.17030715942383 sec.
       1071_0024680_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 63.226845026016235 sec.
       1031_0003144_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 64.27628374099731 sec.
       1031_0003065_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 67.20675206184387 sec.
       1031_0002187_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 66.89972186088562 sec.
       1071_0024766_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 70.05838322639465 sec.
       1071_0024862_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 69.85452389717102 sec.
       1023_0101688_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 78.44429016113281 sec.
       1091_0000064_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 79.59182786941528 sec.
       1061_0120423_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 78.01323103904724 sec.
       1091_0000213_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 78.91283583641052 sec.
       1023_0109590_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 77.48634481430054 sec.
       1071_0248307_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 76.97672176361084 sec.
       1061_0120329_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 80.28094387054443 sec.
       1023_0101846_DE_C1.txt.parsed.txt
       Got getLexFeatures for ONE file in 90.08925795555115 sec.
       1071_0024815_DE_A1.txt.parsed.txt
       Got getLexFeatures for ONE file in 89.52038979530334 sec.
       1061_0120350_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 87.06282496452332 sec.
       1023_0101895_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 87.90472292900085 sec.
       1071_0024799_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 88.89076590538025 sec.
       1091_0000011_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 91.9248149394989 sec.
       1091_0000266_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 96.71859002113342 sec.
       1071_0024823_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 95.9616630077362 sec.
       1061_0120271_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 95.42204093933105 sec.
       1091_0000068_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 93.53404903411865 sec.
       1061_0120491_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 95.40275025367737 sec.
       1061_0120366_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 104.91603112220764 sec.
       1091_0000025_DE_A2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1091_0000025_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.3613278865814209 sec.
       1023_0109591_DE_B2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1023_0109591_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.3356332778930664 sec.
       1061_0120326_DE_B1.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1061_0120326_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.33597683906555176 sec.
       1061_0120312_DE_A2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1061_0120312_DE_A2.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.3209199905395508 sec.
       1023_0101689_DE_B1.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1023_0101689_DE_B1.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.3336319923400879 sec.
       1031_0003173_DE_B2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1031_0003173_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 0.34200310707092285 sec.
       1031_0002091_DE_B2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1031_0002091_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 1.7645833492279053 sec.
       1031_0003052_DE_B2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1031_0003052_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 2.8318111896514893 sec.
       1031_0003225_DE_B2.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1031_0003225_DE_B2.txt.parsed.txt
       Got getLexFeatures for ONE file in 1.4324359893798828 sec.
       1071_0248330_DE_B1.txt.parsed.txt
       Ignoring this text:  /Users/blanchemiret/Workspace/M2_Info/NLP_Ballier/Projet/nlp_cefr_classification/src/../data/DE-Parsed/1071_0248330_DE_B1.txt.parsed.txt
       ```
   
       --> Il s'agit de la fonction `getErrorFeatures`(appelée dans getLexFeatures) qui met autant de temps. C'est le moment où on essaie d'extraire les "error features" donc, voir le `c` des linguistic features dans l'article. Ce qui est donc fait normalement avec le package `language-tool`dont je parle plus haut dans ce doc.
   
       On voit bien ici que le temps par document est problématique, et qu'il croît. N'ayant pas d'autre solution, j'ai laissé tourner toute la nuit, c'est ce qui donne ces logs : au bout d'un moment en fait "il" n'essaie même plus, d'aller chercher les errorfeatures : cf tous les `Ignoring this text`. Dans le code c'est un block try catch qui renvoie cette erreur :
   
       ```
       def getErrorFeatures(conllufilepath, lang):
           numerr = 0
           numspellerr = 0
           try:
               # checker = language_check.LanguageTool(lang)
               checker = language_tool_python.LanguageTool(lang)
               text = makeTextOnly(conllufilepath)
               matches = checker.check(text)
               for match in matches:
                   # if not match.locqualityissuetype == "whitespace":
                   if not match.ruleIssueType == "whitespace":
                       numerr = numerr +1
                       # if match.locqualityissuetype == "typographical" or match.locqualityissuetype == "misspelling":
                       if match.ruleIssueType == "typographical" or match.ruleIssueType == "misspelling":
                           numspellerr = numspellerr +1
           except:
               print("Ignoring this text: ", conllufilepath)
              # numerr = np.nan
              # numspellerr = np.nan
       
           return [numerr, numspellerr]
       ```
   
       DONC
   
       - Si on regarde tous les logs, en gros pour les expériences multilingues, il a extrait les errorfeatures des premiers doc de l'allemand avant d'abandonner (correspond à la stacktrace), sachant qu'il y a + 1000 docs en allemand, j'ai pas compté mais il a dû en traiter environ 100 là 
       - Pour multilingue italien : zéro error features extraites (il a les autres domain features en revanche à priori)
       - Pour les expériences cross-lingue, zéro error features extraites
       - En tchèque, pas de tentative d'extraction d'error features, c'est ce qui est expliqué dans l'article. C'est pour ça que pour ces résultats c'est + proche de ce qui est dans l'article.
   
       Ce qu'il aurait été bien de faire :
   
       - Essayer d'installer la bonne version du JDK et d'avoir la même version de package qu'eux. Donc c'est un peu de notre faute, mais : 
   
       En se remettant du point de vue du projet, qu'est-ce qu'on peut dire ?
   
       - Un reproche général qu'on peut faire pour la reproductibilité : il manque un `requirements.txt`au code original, pour indiquer les versions des packages utilisés, afin de retrouver le même environnement d'exécution. Il aurait aussi été bon de parler des ressources machines utilisées pour les expériences, on a peut-être eu un manque de mémoire vive ici. 