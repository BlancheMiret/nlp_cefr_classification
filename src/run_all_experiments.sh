cd ..

##################################################
############ PREPARATION ENVIRONNEMENT ###########

#pip3 install -r requirements.txt

############################################
############ PREPARATION DONNEES ###########

# téléchargement des données ????
echo "Unziping data..."
unzip data.zip > dev.null

############################################
################ EXPÉRIENCES ###############

cd src

echo
echo "###################################################"
echo "Running all experiments without neural network..."
echo "###################################################"
echo
python3 IdeaPoc.py
echo
echo "###################################################"

echo
echo "###################################################"
echo "Running monolingual neural network experiments..."
echo "###################################################"
echo

for lang in DE IT CZ
do
    echo "###### LANG: ${lang}"
    python3 monolingual_cv.py ../data/${lang}
    echo
done

echo "###################################################"

echo
echo "###################################################"
echo "Running multilingual neural network experiments..."
echo "###################################################"
echo
python3 multi_lingual.py ../data
echo
echo "###################################################"

echo
echo "###################################################"
echo "Running crosslingual neural network experiments without lang..."
echo "###################################################"
echo
python3 multi_lingual_no_langfeat.py ../data
echo
echo "###################################################"
