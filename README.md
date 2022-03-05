# NLP CEFR Classification



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

   - Notes:

     -  script missed a few lines (did not write the output text)

     - 441 files in CZ vs 434 in the delivered code

     - 1033 vs 1029 in DE

     - 813 vs 804 in IT

3. Extract metadata:

   - `$ python3 src/CreateMetadataFile.py`
   - Notes: 
     - OK

4. `ErrorStats.py`

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

- Install `udpipe`