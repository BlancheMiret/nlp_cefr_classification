import os
import collections
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
from sklearn.pipeline import Pipeline

import language_tool_python

SCRIPT_DIR =  os.path.dirname(os.path.realpath(__file__))

seed=1234

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def makePOSsentences(conllufilepath):
    """
    Returns a string which contains one sentence as POS tag sequence per line

    Input:
    Example: "1023_0001416_DE_B2.txt.parsed.txt"

    Output:
    "
    NE NE PROPN CARD CARD NN NN PROPN NN NOUN PROPN PUNCT # one sentence
    NUM NUM PROPN PROPN PROPN PUNCT ...
    "
    """
    fh =  open(conllufilepath)
    everything_POS = []

    pos_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            pos_string = " ".join(pos_sentence) + "\n"
            everything_POS.append(pos_string)
            pos_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            # get UPOS tag from udpipe parsing
            pos_tag = line.split("\t")[3]
            pos_sentence.append(pos_tag)
    fh.close()
    return " ".join(everything_POS)

def makeTextOnly(conllufilepath):
    """
    Returns a string which contains one sentence as POS tag sequence per line

    Output:
    M. Meier Müllergasse 1 12345 Stadt X Internationale Au-pair Vermittlung Bahnofstr .
    101 65185 Wiesbaden Stadt X ...
    """
    fh =  open(conllufilepath)
    allText = []
    this_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            word_string = " ".join(this_sentence) + "\n"
            allText.append(word_string)
            this_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            word = line.split("\t")[1]
            this_sentence.append(word)
    fh.close()
    return " ".join(allText)


def makeDepRelSentences(conllufilepath):
    """
    convert a sentence into this form: nmod_NN_PRON, dobj_VB_NN etc. i.e., each word is replaced by a dep. trigram of that form.
    So full text will look like this instead of a series of words or POS tags:
    root_X_ROOT punct_PUNCT_X case_ADP_PROPN nmod_PROPN_X flat_PROPN_PROPN
     root_PRON_ROOT nsubj_NOUN_PRON case_ADP_PROPN det_DET_PROPN nmod_PROPN_NOUN
     case_ADP_NOUN det_DET_NOUN nummod_NUM_NOUN obl_NOUN_VERB root_VERB_ROOT case_ADP_NOUN det_DET_NOUN obl_NOUN_VERB appos_PROPN_NOUN flat_PROPN_PROPN case_ADP_NOUN obl_NOUN_VERB cc_CCONJ_PART conj_PART_PROPN punct_PUNCT_VERB
     advmod_ADJ_VERB case_ADP_VERB case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB det_DET_NUM obl_NUM_VERB root_VERB_ROOT punct_PUNCT_VERB
     root_PRON_ROOT obj_NOUN_PROPN det_DET_PROPN amod_PROPN_PRON cc_CCONJ_ADV conj_ADV_PROPN cc_CCONJ_ADV punct_PUNCT_PROPN advmod_ADV_PUNCT case_ADP_ADJ advmod_ADV_PUNCT conj_ADV_PROPN amod_PROPN_PRON appos_PROPN_PROPN flat_PROPN_PROPN punct_PUNCT_PROPN
    """
    fh =  open(conllufilepath)
    wanted_features = []
    deprels_sentence = []
    sent_id = 0
    head_ids_sentence = []
    pos_tags_sentence = []
    wanted_sentence_form = []
    id_dict = {}
    id_dict['0'] = "ROOT"
    for line in fh:
        if line == "\n":
            for rel in deprels_sentence:
                wanted = rel + "_" + pos_tags_sentence[deprels_sentence.index(rel)] + "_" +id_dict[head_ids_sentence[deprels_sentence.index(rel)]]
                wanted_sentence_form.append(wanted)
                #Trigrams of the form case_ADP_PROPN, flat_PROPN_PROPN etc.

            wanted_features.append(" ".join(wanted_sentence_form) + "\n")
            deprels_sentence = []
            pos_tags_sentence = []
            head_ids_sentence = []
            wanted_sentence_form = []
            sent_id = sent_id+1
            id_dict = {}
            id_dict['0'] = "root" #LOWERCASING. Some problem with case of features in vectorizer.

        elif not line.startswith("#") and "-" not in line.split("\t")[0]:
            fields = line.split("\t")
            pos_tag = fields[3]
            deprels_sentence.append(fields[7])
            id_dict[fields[0]] = pos_tag
            pos_tags_sentence.append(pos_tag)
            head_ids_sentence.append(fields[6])
    fh.close()
    return " ".join(wanted_features)

def getLexFeatures(conllufilepath,lang, err):
    """
    As described in Lu, 2010: http://onlinelibrary.wiley.com/doi/10.1111/j.1540-4781.2011.01232_1.x/epdf
    Lexical words (N_lex: all open-class category words in UD (ADJ, ADV, INTJ, NOUN, PROPN, VERB)
    All words (N)
    Lex.Density = N_lex/N
    Lex. Variation = Uniq_Lex/N_Lex
    Type-Token Ratio = Uniq_words/N
    Verb Variation = Uniq_Verb/N_verb
    Noun Variation
    ADJ variation
    ADV variation
    Modifier variation
    """
    fh =  open(conllufilepath)
    ndw = [] #To get number of distinct words
    ndn = [] #To get number of distinct nouns - includes propn
    ndv = [] #To get number of distinct verbs
    ndadj = []
    ndadv = []
    ndint = []
    numN = 0.0 #INCL PROPN
    numV = 0.0
    numI = 0.0 #INTJ
    numADJ = 0.0
    numADV = 0.0
    numIntj = 0.0
    total = 0.0
    numSent = 0.0
    for line in fh:
        if not line == "\n" and not line.startswith("#"):
            fields = line.split("\t")
            word = fields[1]
            pos_tag = fields[3]
            if word.isalpha():
                if not word in ndw:
                    ndw.append(word)
                if pos_tag == "NOUN" or pos_tag == "PROPN":
                    numN = numN +1
                    if not word in ndn:
                        ndn.append(word)
                elif pos_tag == "ADJ":
                    numADJ = numADJ+1
                    if not word in ndadj:
                        ndadj.append(word)
                elif pos_tag == "ADV":
                    numADV = numADV+1
                    if not word in ndadv:
                        ndadv.append(word)
                elif pos_tag == "VERB":
                    numV = numV+1
                    if not word in ndv:
                        ndv.append(word)
                elif pos_tag == "INTJ":
                    numI = numI +1
                    if not word in ndint:
                        ndint.append(word)
        elif line == "\n":
            numSent = numSent +1
        total = total +1

    if err:
        try:
            error_features = getErrorFeatures(conllufilepath,lang)
        except:
            print("Ignoring file:",conllufilepath)
            error_features = [0,0]
    else:
        error_features = ['NA', 'NA']

    #Total Lexical words i.e., tokens
    nlex = float(numN + numV + numADJ + numADV + numI)

    #Distinct Lexical words i.e., types
    dlex = float(len(ndn) + len(ndv) + len(ndadj) + len(ndadv) + len(ndint))
    result = [total, round(total/numSent,2), round(len(ndw)/total,2), round(nlex/total,2), round(dlex/nlex,2), round(len(ndv)/nlex,2), round(len(ndn)/nlex,2),
              round(len(ndadj)/nlex,2), round(len(ndadv)/nlex,2), round((len(ndadj) + len(ndadv))/nlex,2),error_features[0], error_features[1]]

    #remove last two features - they are error features which are NA for cz
    if not err:
       return result[:-2]
    else:
       return result

def getErrorFeatures(conllufilepath, lang):
    """
    Num. Errors. NumSpellErrors
    May be other error based features can be added later.
    """
    numerr = 0
    numspellerr = 0
    try:
        checker = language_tool_python.LanguageTool(lang)
        text = makeTextOnly(conllufilepath)
        matches = checker.check(text)
        for match in matches:
            if not match.ruleIssueType == "whitespace":
                numerr = numerr +1
                if match.ruleIssueType == "typographical" or match.ruleIssueType == "misspelling":
                    numspellerr = numspellerr +1
    except:
        print("Ignoring this text: ", conllufilepath)
    return [numerr, numspellerr]


def getScoringFeatures(dirpath,lang,err):
    """
    get features that are typically used in scoring models using getErrorFeatures and getLexFeatures functions.
    err - indicates whether or not error features should be extracted. Boolean. True/False
    """
    files = os.listdir(dirpath)
    fileslist = []
    featureslist = []
    pt0 = time.time()
    for filename in files:
        if filename.endswith(".txt") and filename != ".parsed.txt":
            pt = time.time()
            features_for_this_file = getLexFeatures(os.path.join(dirpath,filename),lang,err)
            fileslist.append(filename)
            featureslist.append(features_for_this_file)
    return fileslist, featureslist


def getLangData(dirpath, option):
    """
    Function to get n-gram like features for Word, POS, and Dependency representations
    option takes: word, pos, dep. default is word

    Ex : getLangData("../data/DE-Parsed", option="pos")

    Output
    fileslist = [ # que ceux en .txt
        "1023_0001416_DE_B2.txt.parsed.txt",
        ...
    ]
    posversionslist = [doc1, doc2, ...]

    1 doc =
        - "pos": "NN PROPN NE NE PROPN CARD CARD NN NN PROPN NN NOUN PROPN PUNCT..." # 1 saut de ligne dans la str = nvlle phrase
        - "dep": "root_X_ROOT punct_PUNCT_X case_ADP_PROPN nmod_PROPN_X flat_PROPN_PROPN ..."
        - else: "M. Meier Müllergasse 1 12345 Stadt X Internationale Au-pair Vermittlung Bahnofstr ."
    """
    files = os.listdir(dirpath)
    fileslist = []
    posversionslist = []
    for filename in files:
        if filename.endswith(".txt") and filename != ".parsed.txt":
            if option == "pos":
                pos_version_of_file = makePOSsentences(os.path.join(dirpath,filename))
            elif option == "dep":
                pos_version_of_file = makeDepRelSentences(os.path.join(dirpath,filename))
            else:
                pos_version_of_file = makeTextOnly(os.path.join(dirpath,filename))
            fileslist.append(filename)
            posversionslist.append(pos_version_of_file)
    return fileslist, posversionslist


def getcatlist(filenameslist):
    """
    Get categories from filenames  -Classification
    """
    result = []
    for name in filenameslist:
        #result.append(name.split("_")[3].split(".txt")[0])
        result.append(name.split(".txt")[0].split("_")[-1])
    return result


def getlangslist(filenameslist):
    """
    Get langs list from filenames - to use in megadataset classification
    """
    result = []
    for name in filenameslist:
        if "_DE_" in name:
           result.append("de")
        elif "_IT_" in name:
           result.append("it")
        else:
           result.append("cz")
    return result


def train_onelang_classification(train_labels,train_data,labelascat=False, langslist=None):
    """
    Training on one language data, Stratified 10 fold CV
    """
    # Extract ngrams
    vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10)
    classifier = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed)
    k_fold = StratifiedKFold(10,random_state=seed,shuffle=True)

    train_vector = vectorizer.fit_transform(train_data).toarray()

    if labelascat and len(langslist) > 1: # pour le multimodel, on introduit la langue dans les vecteurs des doc IF labelascat
        train_vector = enhance_features_withcat(train_vector,language=None,langslist=langslist)

    print("Running cross val...")
    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)

    print("Confusion matrix and weighted f1 score:")
    print(confusion_matrix(train_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
    print(f1_score(train_labels,predicted,average='weighted'))


def combine_features(train_labels,train_sparse,train_dense):
    """
    Combine features like this: get probability distribution over categories with n-gram features. Use that distribution as a feature set concatenated with the domain features - one way to combine sparse and dense feature groups.
    Just testing this approach here.
    """
    k_fold = StratifiedKFold(10,random_state=seed, shuffle=True)
    vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 2000)
    train_vector = vectorizer.fit_transform(train_sparse).toarray()
    classifier = RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed)

    #Get probability distribution for classes.
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, method="predict_proba")

    #Use those probabilities as the new featureset.
    new_features = []
    for i in range(0,len(predicted)):
       temp = list(predicted[i]) + list(train_dense[i])
       new_features.append(temp)

    #predict with new features
    new_predicted = cross_val_predict(classifier, new_features, train_labels, cv=k_fold)
    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)

    print("Scores:")
    print("Acc: " ,str(sum(cross_val)/float(len(cross_val))))
    print("F1: ", str(f1_score(train_labels,new_predicted,average='weighted')))


def cross_lang_testing_classification(train_labels,train_data, test_labels, test_data):
    """
    train on one language and test on another, classification
    """
    vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10) #, max_features = 2000
    classifier = RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed)

    text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
    text_clf.fit(train_data,train_labels)

    predicted = text_clf.predict(test_data)

    print("Confusion table and weighted F1 score:")
    print(confusion_matrix(test_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
    print(f1_score(test_labels,predicted,average='weighted'))


def singleLangClassificationWithoutVectorizer(train_vector,train_labels):
    """
    Single language, 10 fold cv for domain features - i.e., non n-gram features.
    """
    k_fold = StratifiedKFold(10,random_state=seed, shuffle=True)
    classifier = RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed)

    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
    print("Confusion table and weighted F1 score:")
    print(confusion_matrix(train_labels, predicted))
    print(f1_score(train_labels,predicted,average='weighted'))

def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    """
    Cross lingual classification evaluation for non ngram features
    """
    classifier = RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed)

    classifier.fit(train_vector,train_labels)
    predicted = classifier.predict(test_vector)

    print("Confusion table and weighted F1 score:")
    print(confusion_matrix(test_labels,predicted))
    print(f1_score(test_labels,predicted,average='weighted'))


def crossLangRegressionWithoutVectorizer(train_vector, train_scores, test_vector, test_scores):
    """
    Cross lingual regression evaluation for non ngram features
    """
    print("CROSS LANG EVAL")
    regressors = [RandomForestRegressor()]
    k_fold = StratifiedKFold(10,random_state=seed, shuffle=True)
    for regressor in regressors:
        cross_val = cross_val_score(regressor, train_vector, train_scores, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(regressor, train_vector, train_scores, cv=k_fold)
        predicted[predicted < 0] = 0
        print("Cross Val Results: ")
        print(regEval(predicted,train_scores))
        regressor.fit(train_vector,train_scores)
        predicted =regressor.predict(test_vector)
        predicted[predicted < 0] = 0
        print("Test data Results: ")
        print(regEval(predicted,test_scores))


def enhance_features_withcat(features,language=None,langslist=None):
   """
   1dd label features as one hot vector.
   Forma: de - 1 0 0, it - 0 1 0, cz - 0 0 1
   """
   addition = {'de':[1,0,0], 'it': [0,1,0], 'cz': [0,0,1]}
   if language:
        for i in range(0,len(features)):
           features[i].extend(addition[language])
        return features
   if langslist:
        features = np.ndarray.tolist(features)
        for i in range(0,len(features)):
           features[i].extend(addition[langslist[i]])
        return features


def do_mega_multilingual_model_all_features(lang1path,lang1,lang2path,lang2,lang3path,lang3,modelas, setting,labelascat):
   """
   Goal: combine all languages data into one big model
   setting options: pos, dep, domain
   labelascat = true, false (to indicate whether to add label as a categorical feature)
   """
   if not setting == "domain":
      lang1files,lang1features = getLangData(lang1path,setting)
      lang1labels = getcatlist(lang1files)
      lang2files,lang2features = getLangData(lang2path,setting)
      lang2labels = getcatlist(lang2files)
      lang3files,lang3features = getLangData(lang3path,setting)
      lang3labels = getcatlist(lang3files)

   else: #i.e., domain features only.
      lang1files,lang1features = getScoringFeatures(lang1path,lang1,False)
      lang1labels = getcatlist(lang1files)
      lang2files,lang2features = getScoringFeatures(lang2path,lang2,False)
      lang2labels = getcatlist(lang2files)
      lang3files,lang3features = getScoringFeatures(lang3path,lang3,False)
      lang3labels = getcatlist(lang3files)

   # megalabels : ["B1", "C2", ...]
   megalabels = lang1labels + lang2labels + lang3labels

   # megalangs : ["de", "de", ..., "it", "it", ..., "cz", ...]
   megalangs = getlangslist(lang1files) + getlangslist(lang2files) + getlangslist(lang3files)

    # megadata = [doc1_de, doc2_de, ..., doc1_it, ... doc1_cz,...] # 1 doc = 1 str
   if labelascat and setting == "domain":
      megadata = enhance_features_withcat(lang1features,"de") + enhance_features_withcat(lang2features,"it") + enhance_features_withcat(lang3features,"cz")
   else:
      megadata = lang1features + lang2features + lang3features

   print("Distribution of labels:", collections.Counter(megalabels))
   if setting == "domain":
      singleLangClassificationWithoutVectorizer(megadata,megalabels)
   else:
      train_onelang_classification(megalabels,megadata,labelascat,megalangs)

"""
this function does cross language evaluation.
takes a language data directory path, and lang code for both source and target languages.
gets all features (no domain features for cz), and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_cross_lang_all_features(sourcelangdirpath,sourcelang,modelas,targetlangdirpath, targetlang):
   #Read source language data
   sourcelangfiles,sourcelangposngrams = getLangData(sourcelangdirpath, "pos")
   sourcelangfiles,sourcelangdepngrams = getLangData(sourcelangdirpath, "dep")

   #Read target language data
   targetlangfiles,targetlangposngrams = getLangData(targetlangdirpath, "pos")
   targetlangfiles,targetlangdepngrams = getLangData(targetlangdirpath, "dep")

   #Get label info
   sourcelanglabels = getcatlist(sourcelangfiles)
   targetlanglabels = getcatlist(targetlangfiles)

   sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,False)
   targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,False)

   print("\n****** With features: POS n-grams")
   cross_lang_testing_classification(sourcelanglabels,sourcelangposngrams, targetlanglabels, targetlangposngrams)

   print("\n****** With features: dependency n-grams")
   cross_lang_testing_classification(sourcelanglabels,sourcelangdepngrams, targetlanglabels, targetlangdepngrams)

   print("\n****** With features: domain features")
   crossLangClassificationWithoutVectorizer(sourcelangdomain,sourcelanglabels,targetlangdomain,targetlanglabels)


"""
this function takes a language data directory path, and lang code,
gets all features, and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_single_lang_all_features(langdirpath,lang):

    langfiles,langwordngrams = getLangData(langdirpath, "word")
    langfiles,langposngrams = getLangData(langdirpath, "pos")
    langfiles,langdepngrams = getLangData(langdirpath, "dep")

    langfiles,langdomain = getScoringFeatures(langdirpath,lang,False)

    print("Extracted all features:")
    langlabels = getcatlist(langfiles)

    print("Class statistics:", collections.Counter(langlabels))

    print("\n****** With Word ngrams:")
    train_onelang_classification(langlabels,langwordngrams)

    print("\n****** With POS ngrams:")
    train_onelang_classification(langlabels,langposngrams)

    print("\n****** With Dep ngrams:")
    train_onelang_classification(langlabels,langdepngrams)

    print("\n****** With Domain features only:")
    singleLangClassificationWithoutVectorizer(langdomain,langlabels)

    print("\n****** With combined: wordngrams + domain")
    combine_features(langlabels,langwordngrams,langdomain)

    print("\n****** With combined: posngrams + domain")
    combine_features(langlabels,langposngrams,langdomain)

    print("\n****** With combined: depngrams + domain")
    combine_features(langlabels,langdepngrams,langdomain)

def main():

    itdirpath = os.path.join(SCRIPT_DIR, "../data/IT-Parsed")
    dedirpath = os.path.join(SCRIPT_DIR, "../data/DE-Parsed")
    czdirpath = os.path.join(SCRIPT_DIR, "../data/CZ-Parsed")

    print("#########################")
    print("# MONOLINGUAL EXPERIMENTS")
    print()

    print(f"##### MONOLINGUAL MODELS - FEATURES: all - LANG : DE #####")
    do_single_lang_all_features(dedirpath,"de")

    print(f"\n##### MONOLINGUAL MODELS - FEATURES: all - LANG : IT #####")
    do_single_lang_all_features(itdirpath,"it")

    print(f"\n##### MONOLINGUAL MODELS - FEATURES: all - LANG : CZ #####")
    do_single_lang_all_features(czdirpath,"cz")

    print()
    print("##########################")
    print("# MULTLINGUAL EXPERIMENTS")
    print()

    for features in ["word", "dep", "domain"]:
        for withlang in [True, False]:
            print(f"##### MULTILINGUAL MODEL - FEATURES: {features} - WITH LANG: {withlang} #####")
            do_mega_multilingual_model_all_features(dedirpath,"de",itdirpath,"it",czdirpath,"cz",modelas="class", setting=features, labelascat=withlang)
            print("\n")

    print()
    print("###########################")
    print("# CROSS-LINGUAL EXPERIMENTS")
    print()

    print(f"##### CROSS-LINGUAL MODELS - FEATURES: all - TEST ON: IT #####")
    do_cross_lang_all_features(dedirpath,"de","class", itdirpath, "it")

    print(f"\n##### CROSS-LINGUAL MODELS - FEATURES: all - TEST ON: CZ #####")
    do_cross_lang_all_features(dedirpath,"de","class", czdirpath, "cz")

if __name__ == "__main__":
   main()
