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

   - Notes : :ok: 

8. multi_lingual_no_langfeat.py

   - Notes: :ok: 
   
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