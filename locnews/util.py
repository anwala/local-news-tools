import glob
import gzip
import hashlib
import json
import logging
import os
import sys

from datetime import datetime
from re import split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from locnews.config import __installpath__

logger = logging.getLogger('us_pulse.us_pulse')

#date - start
def getNowFilename():
    filename = str(datetime.now()).split('.')[0]
    return filename.replace(' ', 'T').replace(':', '-')
#date - end

#url - start
def getStrHash(txt):

    txt = txt.strip()
    if( txt == '' ):
        return ''

    hash_object = hashlib.md5(txt.encode())
    return hash_object.hexdigest()
#url - end

def procLogHandler(handler, loggerDets):
    
    if( handler is None ):
        return
        
    if( 'level' in loggerDets ):
        handler.setLevel( loggerDets['level'] )    
        
        if( loggerDets['level'] == logging.ERROR ):
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s :\n%(message)s')
            handler.setFormatter(formatter)

    if( 'format' in loggerDets ):
        
        loggerDets['format'] = loggerDets['format'].strip()
        if( loggerDets['format'] != '' ):
            formatter = logging.Formatter( loggerDets['format'] )
            handler.setFormatter(formatter)

    logger.addHandler(handler)

def setLoggerDets(logger, loggerDets):

    if( len(loggerDets) == 0 ):
        return

    consoleHandler = logging.StreamHandler()

    if( 'level' in loggerDets ):
        logger.setLevel( loggerDets['level'] )
    else:
        logger.setLevel( logging.INFO )

    if( 'file' in loggerDets ):
        loggerDets['file'] = loggerDets['file'].strip()
        
        if( loggerDets['file'] != '' ):
            fileHandler = logging.FileHandler( loggerDets['file'] )
            procLogHandler(fileHandler, loggerDets)

    procLogHandler(consoleHandler, loggerDets)

def setLogDefaults(params):
    
    params['log_dets'] = {}

    if( params['log_level'] == '' ):
        params['log_dets']['level'] = logging.INFO
    else:
        
        logLevels = {
            'CRITICAL': 50,
            'ERROR': 40,
            'WARNING': 30,
            'INFO': 20,
            'DEBUG': 10,
            'NOTSET': 0
        }

        params['log_level'] = params['log_level'].strip().upper()

        if( params['log_level'] in logLevels ):
            params['log_dets']['level'] = logLevels[ params['log_level'] ]
        else:
            params['log_dets']['level'] = logging.INFO
    
    params['log_format'] = params['log_format'].strip()
    params['log_file'] = params['log_file'].strip()

    if( params['log_format'] != '' ):
        params['log_dets']['format'] = params['log_format']

    if( params['log_file'] != '' ):
        params['log_dets']['file'] = params['log_file']

def genericErrorInfo(slug=''):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    
    errMsg = fname + ', ' + str(exc_tb.tb_lineno)  + ', ' + str(sys.exc_info())
    logger.error(errMsg + slug)

    return errMsg

def dumpJsonToFile(outfilename, dictToWrite, indentFlag=True, extraParams=None):

    if( extraParams is None ):
        extraParams = {}

    extraParams.setdefault('verbose', True)

    try:
        outfile = open(outfilename, 'w')
        
        if( indentFlag ):
            json.dump(dictToWrite, outfile, ensure_ascii=False, indent=4)#by default, ensure_ascii=True, and this will cause  all non-ASCII characters in the output are escaped with \uXXXX sequences, and the result is a str instance consisting of ASCII characters only. Since in python 3 all strings are unicode by default, forcing ascii is unecessary
        else:
            json.dump(dictToWrite, outfile, ensure_ascii=False)

        outfile.close()

        if( extraParams['verbose'] ):
            logger.info('\tdumpJsonToFile(), wrote: ' + outfilename)
    except:
        genericErrorInfo('\n\terror: outfilename: ' + outfilename)
        return False

    return True

def getDictFromFile(filename):

    try:

        if( os.path.exists(filename) == False ):
            return {}

        return getDictFromJson( readTextFromFile(filename) )
    except:
        print('\tgetDictFromFile(): error filename', filename)
        genericErrorInfo()

    return {}

def getDictFromJson(jsonStr):

    try:
        return json.loads(jsonStr)
    except:
        genericErrorInfo()

    return {}

def getDictFromJsonGZ(path):

    json = getTextFromGZ(path)
    if( len(json) == 0 ):
        return {}
    return getDictFromJson(json)

def getTextFromGZ(path):
    
    try:
        infile = gzip.open(path, 'rb')
        txt = infile.read().decode('utf-8')
        infile.close()

        return txt
    except:
        genericErrorInfo()

    return ''

def gzipTextFile(path, txt):
    
    try:
        with gzip.open(path, 'wb') as f:
            f.write(txt.encode())
    except:
        genericErrorInfo()

def readTextFromFile(infilename):

    text = ''

    try:
        with open(infilename, 'r') as infile:
            text = infile.read()
    except:
        print('\treadTextFromFile()error filename:', infilename)
        genericErrorInfo()
    
    return text

#text - start

def get_color_txt(txt, ansi_code='91m'):
    
    if( ansi_code == '' ):
        return txt
    return '\033[' + ansi_code + '{}\033[00m'.format(txt)

def word_tokenizer(txt, split_pattern="[^a-zA-Z0-9.'’]"):
    
    txt = txt.replace('\n', ' ')
    tokens = split(split_pattern, txt)
    tokens = [ w for w in tokens if w != '' ]

    return tokens

def isStopword(term, stopWordsDict):

    if( term.strip().lower() in stopWordsDict ):
        return True
    else:
        return False

def getStopwordsSet(frozenSetFlag=False):
    
    stopwords = getStopwordsDict()
    
    if( frozenSetFlag ):
        return frozenset(stopwords.keys())
    else:
        return set(stopwords.keys())

def getStopwordsDict():

    stopwordsDict = {
        "a": True,
        "about": True,
        "above": True,
        "across": True,
        "after": True,
        "afterwards": True,
        "again": True,
        "against": True,
        "all": True,
        "almost": True,
        "alone": True,
        "along": True,
        "already": True,
        "also": True,
        "although": True,
        "always": True,
        "am": True,
        "among": True,
        "amongst": True,
        "amoungst": True,
        "amount": True,
        "an": True,
        "and": True,
        "another": True,
        "any": True,
        "anyhow": True,
        "anyone": True,
        "anything": True,
        "anyway": True,
        "anywhere": True,
        "are": True,
        "around": True,
        "as": True,
        "at": True,
        "back": True,
        "be": True,
        "became": True,
        "because": True,
        "become": True,
        "becomes": True,
        "becoming": True,
        "been": True,
        "before": True,
        "beforehand": True,
        "behind": True,
        "being": True,
        "below": True,
        "beside": True,
        "besides": True,
        "between": True,
        "beyond": True,
        "both": True,
        "but": True,
        "by": True,
        "can": True,
        "can\'t": True,
        "cannot": True,
        "cant": True,
        "co": True,
        "could not": True,
        "could": True,
        "couldn\'t": True,
        "couldnt": True,
        "de": True,
        "describe": True,
        "detail": True,
        "did": True,
        "do": True,
        "does": True,
        "doing": True,
        "done": True,
        "due": True,
        "during": True,
        "e.g": True,
        "e.g.": True,
        "e.g.,": True,
        "each": True,
        "eg": True,
        "either": True,
        "else": True,
        "elsewhere": True,
        "enough": True,
        "etc": True,
        "etc.": True,
        "even": True,
        "even though": True,
        "ever": True,
        "every": True,
        "everyone": True,
        "everything": True,
        "everywhere": True,
        "except": True,
        "for": True,
        "former": True,
        "formerly": True,
        "from": True,
        "further": True,
        "get": True,
        "go": True,
        "had": True,
        "has not": True,
        "has": True,
        "hasn\'t": True,
        "hasnt": True,
        "have": True,
        "having": True,
        "he": True,
        "hence": True,
        "her": True,
        "here": True,
        "hereafter": True,
        "hereby": True,
        "herein": True,
        "hereupon": True,
        "hers": True,
        "herself": True,
        "him": True,
        "himself": True,
        "his": True,
        "how": True,
        "however": True,
        "i": True,
        "ie": True,
        "i.e": True,
        "i.e.": True,
        "if": True,
        "in": True,
        "inc": True,
        "inc.": True,
        "indeed": True,
        "into": True,
        "is": True,
        "it": True,
        "its": True,
        "it's": True,
        "itself": True,
        "just": True,
        "keep": True,
        "latter": True,
        "latterly": True,
        "less": True,
        "made": True,
        "make": True,
        "may": True,
        "me": True,
        "meanwhile": True,
        "might": True,
        "mine": True,
        "more": True,
        "moreover": True,
        "most": True,
        "mostly": True,
        "move": True,
        "must": True,
        "my": True,
        "myself": True,
        "namely": True,
        "neither": True,
        "never": True,
        "nevertheless": True,
        "next": True,
        "no": True,
        "nobody": True,
        "none": True,
        "noone": True,
        "nor": True,
        "not": True,
        "nothing": True,
        "now": True,
        "nowhere": True,
        "of": True,
        "off": True,
        "often": True,
        "on": True,
        "once": True,
        "one": True,
        "only": True,
        "onto": True,
        "or": True,
        "other": True,
        "others": True,
        "otherwise": True,
        "our": True,
        "ours": True,
        "ourselves": True,
        "out": True,
        "over": True,
        "own": True,
        "part": True,
        "per": True,
        "perhaps": True,
        "please": True,
        "put": True,
        "rather": True,
        "re": True,
        "same": True,
        "see": True,
        "seem": True,
        "seemed": True,
        "seeming": True,
        "seems": True,
        "several": True,
        "she": True,
        "should": True,
        "show": True,
        "side": True,
        "since": True,
        "sincere": True,
        "so": True,
        "some": True,
        "somehow": True,
        "someone": True,
        "something": True,
        "sometime": True,
        "sometimes": True,
        "somewhere": True,
        "still": True,
        "such": True,
        "take": True,
        "than": True,
        "that": True,
        "the": True,
        "their": True,
        "theirs": True,
        "them": True,
        "themselves": True,
        "then": True,
        "thence": True,
        "there": True,
        "thereafter": True,
        "thereby": True,
        "therefore": True,
        "therein": True,
        "thereupon": True,
        "these": True,
        "they": True,
        "this": True,
        "those": True,
        "though": True,
        "through": True,
        "throughout": True,
        "thru": True,
        "thus": True,
        "to": True,
        "together": True,
        "too": True,
        "toward": True,
        "towards": True,
        "un": True,
        "until": True,
        "upon": True,
        "us": True,
        "very": True,
        "via": True,
        "was": True,
        "we": True,
        "well": True,
        "were": True,
        "what": True,
        "whatever": True,
        "when": True,
        "whence": True,
        "whenever": True,
        "where": True,
        "whereafter": True,
        "whereas": True,
        "whereby": True,
        "wherein": True,
        "whereupon": True,
        "wherever": True,
        "whether": True,
        "which": True,
        "while": True,
        "whither": True,
        "who": True,
        "whoever": True,
        "whole": True,
        "whom": True,
        "whose": True,
        "why": True,
        "will": True,
        "with": True,
        "within": True,
        "without": True,
        "would": True,
        "yet": True,
        "you": True,
        "your": True,
        "yours": True,
        "yourself": True,
        "yourselves": True
    }
    
    return stopwordsDict

def get_non_specific_beats():
    
    payload = {'domains': {}}    
    
    res_path = __installpath__ + 'Resources/non_specific_beats/'
    all_beats = readTextFromFile(res_path + 'all.txt')
    payload['all'] = set(all_beats.strip().splitlines())
    
    for f in glob.glob(res_path + 'domains/*.txt'):
        domain = f.split('/')[-1].replace('.txt', '')
        beats = readTextFromFile(f)
        payload['domains'][domain] = set(beats.strip().splitlines())

    return payload

def get_doc_lst_pos_maps(doc_lst, rm_text=True):

    new_doc_list = []
    pos_id_mapping = {}

    try:

        for i in range( len(doc_lst) ):

            d = doc_lst[i]
            d.setdefault('id', i)
            pos_id_mapping[i] = {'id': d['id']}

            #transfer other properties - start
            for ky, val in d.items():

                if( rm_text is True ):
                    if( ky == 'text' ):
                        continue

                pos_id_mapping[i][ky] = val
            #transfer other properties - end

            new_doc_list.append( d['text'] )
    
    except:
        genericErrorInfo()
        return [], {}

    return new_doc_list, pos_id_mapping

def map_tf_mat_to_doc_ids(payload, pos_id_mapping):

    #see get_tf_matrix().payload for payload's structure
    if( 'tf_matrix' not in payload and 'top_ngrams' not in payload ):
        return {}

    if( 'per_doc' not in payload['top_ngrams'] ):
        return {}

    tf_matrix = []
    top_ngrams_per_doc = []
    for pos, doc_dct in pos_id_mapping.items():
        if( pos < len(payload['tf_matrix']) ):

            tf_matrix.append({
                'id': doc_dct['id'], 
                'tf_vector': payload['tf_matrix'][pos]
            })

            top_ngrams_per_doc.append({
                'id': doc_dct['id'],
                'ngrams': payload['top_ngrams']['per_doc'][pos]
            })

            #transfer other properties - start
            for ky, val in doc_dct.items():
                tf_matrix[-1][ky] = val
                top_ngrams_per_doc[-1][ky] = val
            #transfer other properties - end


    #special case for (optional, see: get_tf_matrix().add_normalized_tf_matrix) tf_matrix_normalized - start
    for opt in ['tf_matrix_normalized', 'tf_idf_matrix']:
        if( opt in payload ):
            
            opt_vect = []
            for pos, doc_dct in pos_id_mapping.items():
                if( pos < len(payload[opt]) ):

                    opt_vect.append({
                        'id': doc_dct['id'], 
                        'tf_vector': payload[opt][pos]
                    })

                    #transfer other properties - start
                    for ky, val in doc_dct.items():
                        opt_vect[-1][ky] = val
                    #transfer other properties - end

            payload[opt] = opt_vect
    #special case for (optional, see: get_tf_matrix().add_normalized_tf_matrix) tf_matrix_normalized - end

    payload['tf_matrix'] = tf_matrix
    payload['top_ngrams']['per_doc'] = top_ngrams_per_doc

    return payload

def get_tf_matrix(doc_lst, n, tf_mat=None, vocab=None, token_pattern=r'(?u)\b[a-zA-Z\'\’-]+[a-zA-Z]+\b|\d+[.,]?\d*', **kwargs):

    if( len(doc_lst) == 0 ):
        return {}

    kwargs.setdefault('rm_doc_text', True)

    if( isinstance(doc_lst[0], dict) ):
        doc_lst, pos_id_mapping = get_doc_lst_pos_maps( doc_lst, rm_text=kwargs['rm_doc_text'] )
    else:
        pos_id_mapping = {}

    if( len(doc_lst) == 0 ):
        return {}

    kwargs.setdefault('set_top_ngrams', True)
    kwargs.setdefault('lowercase', True)
    kwargs.setdefault('add_all_docs', False)
    kwargs.setdefault('add_normalized_tf_matrix', 'l1')
    kwargs.setdefault('add_tf_idf_matrix', 'l2')
    kwargs.setdefault('min_df', 1)
    kwargs.setdefault('stopwords', getStopwordsSet())
    
    #if payload changes, update map_tf_mat_to_doc_ids()
    #also update "convert types for JSON serialization" 
    payload = {
        'tf_matrix': [],
        'tf_matrix_normalized': [],
        'tf_idf_matrix': [],
        'vocab': [],
        'top_ngrams': {
            'per_doc': [],
            'all_docs': []
        },
        'token_pattern': token_pattern
    }
    
    try:
        count_vectorizer = CountVectorizer(
            stop_words=kwargs['stopwords'], 
            token_pattern=token_pattern, 
            ngram_range=(n, n),
            lowercase=kwargs['lowercase'],
            min_df=kwargs['min_df']
        )

        if( tf_mat is not None and vocab is not None ):
            payload['tf_matrix'] = tf_mat
            payload['vocab'] = vocab
        else:
            payload['tf_matrix'] = count_vectorizer.fit_transform(doc_lst).toarray()
            payload['vocab'] = count_vectorizer.get_feature_names()


        if( kwargs['add_normalized_tf_matrix'] != '' ):
            payload['tf_matrix_normalized'] = normalize(payload['tf_matrix'], norm=kwargs['add_normalized_tf_matrix'], axis=1)

        if( kwargs['add_tf_idf_matrix'] != '' ):
            tfidf = TfidfTransformer( norm=kwargs['add_tf_idf_matrix'] )
            tfidf.fit(payload['tf_matrix'])
            payload['tf_idf_matrix'] = tfidf.transform(payload['tf_matrix']).toarray()

    

        #convert types for JSON serialization - start
        for opt in ['tf_matrix', 'tf_matrix_normalized', 'tf_idf_matrix']:

            if( opt not in payload ):
                continue

            payload[opt] = list( payload[opt] )
            for i in range( len(payload[opt]) ):
                payload[opt][i] = [ float(a) for a in payload[opt][i] ]

        #convert types for JSON serialization - end
    except:
        genericErrorInfo()
        return {}

    if( kwargs['set_top_ngrams'] is False ):
        return payload

    all_docs_tf = {}
    all_docs_total_tf = 0
    for i in range( len(payload['tf_matrix']) ):
        
        total_tf = sum(payload['tf_matrix'][i])
        if( total_tf == 0 ):
            total_tf = -1

        single_doc_tf = [ {'term': v, 'term_freq': int(payload['tf_matrix'][i][j]), 'term_rate': payload['tf_matrix'][i][j]/total_tf} for (j, v) in enumerate(payload['vocab']) if payload['tf_matrix'][i][j] != 0 ]
        single_doc_tf = sorted( single_doc_tf, key=lambda i: i['term_freq'], reverse=True )
        
        payload['top_ngrams']['per_doc'].append( single_doc_tf )


        if( kwargs['add_all_docs'] is True ):
            for tf_dct in single_doc_tf:
                all_docs_tf.setdefault( tf_dct['term'], 0 )
                all_docs_tf[ tf_dct['term'] ] += int(tf_dct['term_freq'])
                all_docs_total_tf += int(tf_dct['term_freq'])

    
    if( kwargs['add_all_docs'] is True ):

        if( all_docs_total_tf == 0 ):
            all_docs_total_tf = -1

        payload['top_ngrams']['all_docs'] = sorted( all_docs_tf.items(), key=lambda x: x[1], reverse=True )
        payload['top_ngrams']['all_docs'] = [ {'term': t[0], 'term_freq': t[1], 'term_rate': t[1]/all_docs_total_tf} for t in payload['top_ngrams']['all_docs'] ]

    if( len(pos_id_mapping) != 0 ):
        payload = map_tf_mat_to_doc_ids(payload, pos_id_mapping)

    return payload

#text - end

#set - start

def obsolete_jaccard_set_pair(first_set, second_set):

    intersection = float(len(first_set & second_set))
    union = len(first_set | second_set)

    if( union != 0 ):
        return intersection/union
    else:
        return 0
        
def obsolete_overlap_set_pair(first_set, second_set):

    intersection = float(len(first_set & second_set))
    minimum = min(len(first_set), len(second_set))

    if( minimum != 0 ):
        return intersection/minimum
    else:
        return 0

def obsolete_weighted_jaccard_overlap_sim(first_set, second_set, jaccard_weight):

    if( jaccard_weight > 1 ):
        jaccard_weight = 1
    elif( jaccard_weight < 0 ):
        jaccard_weight = 0

    overlap_weight = 1 - jaccard_weight

    jaccard_weight = jaccard_weight * jaccard_set_pair(first_set, second_set)
    overlap_weight = overlap_weight * overlap_set_pair(first_set, second_set)

    return jaccard_weight + overlap_weight
#set - end