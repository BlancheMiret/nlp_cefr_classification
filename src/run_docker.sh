#!/bin/bash

##################################################
############ PREPARATION ENVIRONNEMENT ###########

#pip3 install -r requirements.txt

############################################
############ PREPARATION DONNEES ###########

cd ..
echo "Unziping data..."
unzip data.zip > /dev/null
cd src

############################################
################ EXPÉRIENCES ###############

IdeaPOC () {
    echo
    echo "###################################################"
    echo "Running all experiments without neural network..."
    echo "###################################################"
    echo
    python3 IdeaPOC.py
    echo
    echo "###################################################"
}

monolingual_cv () {
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
}

multi_lingual () {
    echo
    echo "###################################################"
    echo "Running multilingual neural network experiments..."
    echo "###################################################"
    echo
    python3 multi_lingual.py ../data
    echo
    echo "###################################################"
}

multi_lingual_no_langfeat () {
    echo
    echo "###################################################"
    echo "Running crosslingual neural network experiments without lang..."
    echo "###################################################"
    echo
    python3 multi_lingual_no_langfeat.py ../data
    echo
    echo "###################################################"
}

echo "###################################################"
echo "#                      MENU                       #"
echo "###################################################"
PS3='Please enter your choice: '
options=("Option 1 : All experiments" \
"Option 2 : Experiments without neural network" \
"Option 3 : Monolingual neural network experiments" \
"Option 4 : Multilingual neural network experiments" \
"Option 5 : Crosslingual neural network experiments without lang" \
"Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Option 1 : All experiments")
            IdeaPOC
            monolingual_cv
            multi_lingual
            multi_lingual_no_langfeat
            ;;
        "Option 2 : Experiments without neural network")
            IdeaPOC 
            ;;
        "Option 3 : Monolingual neural network experiments")
            monolingual_cv
            ;;
        "Option 4 : Multilingual neural network experiments")
            multi_lingual
            ;;
        "Option 5 : Crosslingual neural network experiments without lang")
            multi_lingual_no_langfeat
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done