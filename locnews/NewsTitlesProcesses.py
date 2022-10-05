import itertools
import logging
import os
import re

from NwalaTextUtils.textutils import getDedupKeyForURI
from NwalaTextUtils.textutils import parallelGetTxtFrmURIs
from NwalaTextUtils.textutils import updateLogger
from sgsuite.ClusterNews import ClusterNews

from locnews.config import __installpath__
from locnews.NewsTitles import NewsTitles

from locnews.util import dumpJsonToFile
from locnews.util import getDictFromFile
from locnews.util import getStrHash
from locnews.util import get_color_txt
from locnews.util import get_non_specific_beats
from locnews.util import genericErrorInfo
from locnews.util import getNowFilename
from locnews.util import get_tf_matrix
from locnews.util import getStopwordsSet

logger = logging.getLogger('us_pulse.us_pulse')

def naive_clust_news( media_grp, sim_coeff=0.3, jaccard_weight=0.3, min_avg_deg=3, min_uniq_src_count=3 ):

    logger.info('\nnaive_clust_news() - start')

    if( len(media_grp) == 0 ):
        return {}

    media_grp_dedup = []
    uri_dedup_set = set()

    for i in range( len(media_grp) ):
        
        uri_ky = getDedupKeyForURI( media_grp[i]['link'] )  
        if( uri_ky in uri_dedup_set ):
            continue
        uri_dedup_set.add(uri_ky)
        media_grp_dedup.append( media_grp[i] )
        

    #run news clustering algorithm
    graph_stories = ClusterNews( media_grp_dedup, sim_metric='weighted-jaccard-overlap', min_sim=sim_coeff, jaccard_weight=jaccard_weight )
    nodes = graph_stories.cluster_news()

    #annotate nodes so it can be visualized at: http://storygraph.cs.odu.edu/graphs/polar-media-consensus-graph/
    nodes = ClusterNews.annotate(nodes, min_avg_deg=min_avg_deg, min_uniq_src_count=min_uniq_src_count)

    logger.info('naive_clust_news() - end')
    return nodes

def cluster_local_news_titles( media_payload, method='naive', sim_coeff=0.3, jaccard_weight=0.3 ):

    media_grp_clusts = { 'media_group_clusters': [] }
    if( 'media_groups' not in media_payload ):
        return media_grp_clusts

    logger.info('\nnaive_cluster_local_news_titles():')

    for grp in media_payload['media_groups']:
    
        grp_count = len( media_payload['media_groups'][grp] )
        logger.info(f'\tclustering grp: "{grp}" with {grp_count} news titles')

        if( method == 'naive' ):
            nodes = naive_clust_news( media_payload['media_groups'][grp]['links'], sim_coeff=sim_coeff, jaccard_weight=jaccard_weight )

        if( len(nodes) != 0 ):
            media_grp_clusts['media_group_clusters'].append({
                'group_name': grp,
                'graph': nodes
            })

    return media_grp_clusts


def get_news_beats_stats(links, cursor_loc, word_count):

    if( len(links) == 0 ):
        return []

    beat_freq = {}
    for i in range( len(links) ):
            
        l = links[i]
        if( len(l['beat_candidates']) == 0 ):
            continue

        if( len(l['beat_candidates']) > 1 ):
            if( cursor_loc == 'back' ):
                l['beat_candidates'].reverse()

        merged_beats = l['beat_candidates'][:word_count]
        beats_key = '-'.join([b['beat'] for b in merged_beats])
        
        beat_freq.setdefault(beats_key, {'freq': 0, 'beat_candidates': merged_beats, 'link_indices': []})
        beat_freq[beats_key]['freq'] += 1
        beat_freq[beats_key]['link_indices'].append(i)

    beat_freq = sorted(beat_freq.items(), key=lambda x: x[1]['freq'], reverse=True)
    beat_freq = [ b[1] for b in beat_freq]
    return beat_freq

def abbreviate_str(txt, max_len=10):
    
    txt_len = len(txt)
    if( txt_len > max_len ):
        txt = txt[:max_len-2] + '..'
    else:
        txt = txt + ' ' * (max_len - txt_len)

    return txt

def fmt_beat(beat, beat_freq, beat_color_maps, max_len=10, details=None):
    
    beat_color = ''
    if( beat in beat_color_maps ):
        beat_color = beat_color_maps[beat]

    beat = abbreviate_str(beat, max_len)
    beat = get_color_txt(beat, ansi_code=beat_color)
    underline_start = '\033[4m'
    underline_end = '\033[0m'
    
    if( details is not None ):
        beat = underline_start + beat + underline_end

    beat_freq = '{:.2f}'.format(beat_freq).replace('0.', '.').replace('1.00', '1.0')
    beat = beat + ' (' + beat_freq + ')'

    return beat

def present_news_beats(media_payload, all_beats_df, word_count, top_k_beats, **kwargs):

    if( len(media_payload) == 0 ):
        return {}

    kwargs.setdefault('max_beat_len', 10)
    kwargs.setdefault('max_city_len', 15)
    kwargs.setdefault('max_media_len', 20)
    kwargs.setdefault('colors', ['91m', '36m', '33m'])

    colors = kwargs['colors']
    all_beats_df = sorted(all_beats_df.items(), key=lambda x: x[1], reverse=True)
    all_beats_df = all_beats_df[:len(colors)]
    
    beat_color_maps = {}
    for i in range( len(colors) ):
        beat = all_beats_df[i][0]
        beat_color_maps[beat] = colors[i]


    blank_col_count = {}
    beat_report = {}
    for med, links in media_payload['media_groups'].items():    
        
        if( len(links['links']) == 0 ):
            beat_report[med] = []
            continue
        
        single_med_beats = []
        total_beat_freq = sum([ b['freq'] for b in links['beats'] ])

        for beat in links['beats']:
            #['author     (1.0)', '----------------']
            #beat: {'freq': 9, 'beat_candidates': ['news']}: 1 word
            #beat: {'freq': 9, 'beat_candidates': ['news', 'sports']}: 2 words
            fmted_beats = [ fmt_beat(b['beat'], beat['freq']/total_beat_freq, beat_color_maps, max_len=kwargs['max_beat_len'], details=b['details']) for b in beat['beat_candidates'] ]
            
            #this is to ensure all formatted beats are of fixed list length
            fmted_beats += [ ' ' * (kwargs['max_beat_len'] + 6) ] * (word_count - len(fmted_beats))#+6: space(.xx)
            single_med_beats += fmted_beats

        if( len(single_med_beats) < top_k_beats*word_count ):
            #this is to ensure all formatted beats are of fixed list length
            single_med_beats += [ ' ' * (kwargs['max_beat_len'] + 6) ] * ((top_k_beats * word_count) - len(single_med_beats))
        
        single_med_beats = [ '{:<{mw}}'.format(b, mw=kwargs['max_beat_len']) for b in single_med_beats ]
        beat_report[med] = single_med_beats

        for i in range( len(single_med_beats) ):
            blank_col_count.setdefault(i, 0)
            if( single_med_beats[i].strip() == '' ):
                blank_col_count[i] += 1


    #strip all blank columns
    header_template = []
    for col, blnk_count in blank_col_count.items():
        if( blnk_count == len(beat_report) ):
            for med in beat_report:
                beat_report[med][col] = beat_report[med][col].strip()
        else:
            header_template.append('')

    #remove all blank columns
    for med in beat_report:
        beat_report[med] = [c for c in beat_report[med] if c != '']


    #addtional formatting for beats
    for med, links in media_payload['media_groups'].items():    
        single_med_beats = beat_report[med]
        single_med_beats = ' '.join(single_med_beats)
        beat_report[med] = single_med_beats

    
    return {
        'beat_report': beat_report,
        'header_template': header_template,
        'beat_color_maps': beat_color_maps
    }

def get_media_permutation( media_groups, beat_order ):

    if( beat_order not in ['domain', 'joined_beats'] ):
        beat_order = 'joined_beats'

    med_permutation = []
    for media, med_dets in media_groups.items():

        all_beats = [ b for b in med_dets['beats'] ]
        #all_beats = [ ' '.join(b['beat_candidates']) for b in all_beats ]
        all_beats = [ ' '.join([ b['beat'] for b in b['beat_candidates'] ]) for b in all_beats ]
        all_beats = ' '.join(all_beats)
        all_beats = all_beats.strip()

        if( all_beats == '' ):
            all_beats = 'zzzz'

        med_permutation.append({'domain': media, 'joined_beats': all_beats})

    med_permutation = sorted(med_permutation, key=lambda x: x[beat_order])
    med_permutation = [ m['domain'] for m in med_permutation ]

    return med_permutation

def gen_non_specific_beats( media_payload, method='tf_idf' ):
    
    all_beats = {}
    total_freq = 0
    for med, links in media_payload['media_groups'].items():
        
        if( len(links['links']) == 0 ):
            continue

        med_beats = []
        for i in range( len(links['links']) ):
            med_beats += [ bc['beat'] for bc in links['links'][i]['beat_candidates'] ]

        med_beats = set(med_beats)

        for b in med_beats:
            all_beats.setdefault(b, {'freq': 0, 'rate': 0, 'beat': b})
            all_beats[b]['freq'] += 1
            total_freq += 1


    for beat, beat_dets in all_beats.items():
        beat_dets['rate'] = beat_dets['freq']/total_freq
    all_beats = sorted(all_beats.items(), key=lambda x: x[1]['freq'], reverse=True)


    final_beats = []
    print(f'Store final list in {__installpath__}Resources/non_specific_beats/all.txt. For domain-specific beats, create file for domain (e.g., {__installpath__}Resources/non_specific_beats/example.com.txt) and add domain-specific beats.')
    print('Non Specific Beat Candidates:')
    for i in range( len(all_beats) ):
        beat, beat_dets = all_beats[i]
        
        print( '{:>3} {:5} {:<.3f} {}'.format(i, beat_dets['freq'], beat_dets['rate'], beat) )
        
        if( beat_dets['freq'] == 1 ):
            print('\tbreaking at freq = 1')
            break
        final_beats.append(beat_dets)

    return final_beats

def label_non_specific_beats( media_payload, cursor_loc, non_specific_beats_payload ):

    cursor_loc = -1 if cursor_loc == 'back' else 0
    
    non_specific_beats = non_specific_beats_payload['all']
    domain_non_specific_beats = non_specific_beats_payload['domains']

    for med, links in media_payload['media_groups'].items():
        
        if( len(links['links']) == 0 ):
            continue

        for i in range( len(links['links']) ):

            beat_candidates = links['links'][i]['beat_candidates']
            if( len(beat_candidates) == 0 ):
                continue
            
            new_candidates = [ bc for bc in beat_candidates if bc['beat'] not in non_specific_beats ]
            if( med in domain_non_specific_beats ):
                new_candidates = [ bc for bc in new_candidates if bc['beat'] not in domain_non_specific_beats[med] ]
            
            if( len(new_candidates) == 0 ):
                #all new_candidates consisted of non_specific_beats, so pick just 1
                beat_candidates[cursor_loc]['is_specific'] = False
                links['links'][i]['beat_candidates'] = [beat_candidates[cursor_loc]]
            else:
                links['links'][i]['beat_candidates'] = new_candidates

    return media_payload

def cache_link_beats(media, all_med_links, link_indx_map, beats_cache_path):

    media = media.strip()
    beats_cache_path = beats_cache_path.strip()
    if( media == '' or beats_cache_path == '' ):
        return

    try:
        os.makedirs(f'{beats_cache_path}{media}/', exist_ok=True)
    except:
        genericErrorInfo()
        return

    
    for link, indx in link_indx_map.items():
        
        if( 'prev_beat_candidates' not in all_med_links[indx] ):
            continue

        prev_beat_candidates = all_med_links[indx]['prev_beat_candidates']
        beat_candidates = all_med_links[indx]['beat_candidates']
        cf = f'{beats_cache_path}{media}/' + getStrHash(link) + '.json'
        
        cache = {
            'link': link,
            'prev_beat_candidates': prev_beat_candidates,
            'beat_candidates': beat_candidates
        }

        dumpJsonToFile(cf, cache, indentFlag=False)


def tf_resolve_non_specific_beats( media_payload, cursor_loc, word_count, non_specific_beats, **kwargs ):

    beats_cache_path = kwargs.get('beats_cache_path', '')
    beats_cache_path = beats_cache_path if (beats_cache_path.endswith('/') or beats_cache_path == '') else beats_cache_path + '/'

    logger.info('\ntf_resolve_non_specific_beats():')
    logger.info(f'\tbeats_cache_path: {beats_cache_path}')

    all_stopwords = getStopwordsSet() | non_specific_beats['all']

    med_i = 0
    med_count = len(media_payload['media_groups'])
    
    for med, links in media_payload['media_groups'].items():
        
        med_i += 1
        if( len(links['links']) == 0 ):
            continue

        uris_lst = []
        link_indx_map = {}
        for i in range( len(links['links']) ):

            beat_candidates = links['links'][i]['beat_candidates']
            if( len(beat_candidates) != 1 ):
                #see label_non_specific_beats()::if( len(new_candidates) == 0 )'s body
                continue

            if( beat_candidates[0]['is_specific'] is True ):
                continue

            news_link = links['links'][i]['link']

            #attempt to read cache - start
            if( beats_cache_path != '' ):
                cf = getDictFromFile( f'{beats_cache_path}{med}/' + getStrHash(news_link) + '.json' )
                if( 'prev_beat_candidates' in cf and 'beat_candidates' ):
                    links['links'][i]['prev_beat_candidates'] = cf['prev_beat_candidates']
                    links['links'][i]['beat_candidates'] = cf['beat_candidates']
                    continue
            #attempt to read cache - end

            link_indx_map[ news_link ] = i
            uris_lst.append( news_link )
        
        if( len(uris_lst) == 0 ):
            continue


        logger.info( f'\t{med_i} of {med_count}: {med}, {len(uris_lst)} links' )
        doc_lst = parallelGetTxtFrmURIs(uris_lst, cleanHTMLReportFailure=False)
        stopwords = all_stopwords | non_specific_beats['domains'][med] if med in non_specific_beats['domains'] else all_stopwords

        for i in range( len(doc_lst) ):
            
            d = doc_lst[i]
            ori_indx = link_indx_map[ d['uri'] ]
            links['links'][ori_indx]['prev_beat_candidates'] = links['links'][ori_indx].pop('beat_candidates')
            links['links'][ori_indx]['beat_candidates'] = []
            
            d['text'] = d['text'].strip()
            if( d['text'] == '' ):
                continue
            
            tf_mat = get_tf_matrix( doc_lst=[d], n=1, stopwords=stopwords )
            if( len(tf_mat) == 0 ):
                continue

            if( len(tf_mat['top_ngrams']['per_doc']) == 0 ):
                continue

            '''
                To do:
                * don't print too much warning info
            '''
            new_beat_cands = [ n['term'] for n in tf_mat['top_ngrams']['per_doc'][0]['ngrams'][:word_count] ]
            new_beat_cands = [ b for b in new_beat_cands if re.search('[a-zA-Z]', b) is not None ]
            new_beat_cands = NewsTitles.format_news_beats( new_beat_cands, details={'gen_by': 'tf_mat'} )

            if( cursor_loc == 'back' ):
                #order new_beat_cands based on cursor
                #e.g., if cursor_loc is back and new_beat_cands is ['digital', 'ibj', 'edition'], reverse new_beat_cands, so digital would be extracted first when the beats are being extracted, since cursor_loc is back
                new_beat_cands.reverse()

            links['links'][ori_indx]['beat_candidates'] = new_beat_cands

        if( beats_cache_path != '' ):
            cache_link_beats(med, links['links'], link_indx_map, beats_cache_path)
            
    return media_payload

def local_news_beats_explorer( media_payload, cursor_loc='back', word_count=1, top_k_beats=3, **kwargs ):

    if( len(media_payload) == 0 ):
        return media_payload

    kwargs.setdefault('max_beat_len', 10)
    kwargs.setdefault('max_city_len', 15)
    kwargs.setdefault('max_media_len', 20)
    kwargs.setdefault('locations', [])
    kwargs.setdefault('colors', ['91m', '36m', '33m'])
    kwargs.setdefault('order_beats_by', 'beats')
    kwargs.setdefault('beats_resolver', '')
    kwargs.setdefault('beats_cache_path', '')
    user_stopwords = kwargs.get('add_beats_stopword', set())

    non_specific_beats = get_non_specific_beats()
    non_specific_beats['all'] = non_specific_beats['all'] | user_stopwords
    media_payload = label_non_specific_beats( media_payload, cursor_loc, non_specific_beats )

    if( kwargs['beats_resolver'] == 'tf' ):
        media_payload = tf_resolve_non_specific_beats( media_payload, cursor_loc, word_count, non_specific_beats, beats_cache_path=kwargs['beats_cache_path'] )

    word_count = 1 if word_count < 1 else word_count
    top_k_beats = 3 if top_k_beats < 1 else top_k_beats

    print('\nlocal_news_beats_explorer():')
    print('\tcursor_loc :', cursor_loc)
    print('\tword_count :', word_count)
    print('\ttop_k_beats:', top_k_beats)
    print('\tuser_stopwords:', user_stopwords)
    print()

    '''
        cursor_loc: front or back
        word_count: 1
        top_k_beats: k
    '''
    all_beats_df = {}
    for med, links in media_payload['media_groups'].items():
        
        if( len(links['links']) == 0 ):
            continue
        
        beat_freq = get_news_beats_stats( links['links'], cursor_loc, word_count )    
        links['beats'] = beat_freq[:top_k_beats]

        for beat in links['beats']:
            for b in beat['beat_candidates']:
                b = b['beat']
                all_beats_df.setdefault(b, 0)
                all_beats_df[b] += 1
   

    btrp = present_news_beats(media_payload, all_beats_df, word_count, top_k_beats, **kwargs)
    btrp['header_template'] = [ '{:<{mw}}'.format(i+1, mw=kwargs['max_beat_len'] + 6) for i in range(len(btrp['header_template'])) ]#+6 because of space(.xx)
    btrp['header_template'] = ' '.join(btrp['header_template']) + f' | ST' + ' | {} | {} | {}'.format( abbreviate_str('City', kwargs['max_city_len']), abbreviate_str('Media', kwargs['max_media_len']), abbreviate_str('Domain', kwargs['max_media_len']) )
    btrp_len = len(btrp['beat_report'])
    med_permutation = get_media_permutation( media_payload['media_groups'], kwargs['order_beats_by'] )

    print(f'\nTop {top_k_beats} beats of {btrp_len} local media from', kwargs['locations'], 'at', getNowFilename())
    print(' {:>5}'.format(' '), btrp['header_template'])#' {:>5}' because of print(' {:>5}'.format(i),...below
    print(' {:>5}'.format(' '), '-'*len(btrp['header_template']))
    
    for i in range( len(med_permutation) ):
        
        med = med_permutation[i]
        links = media_payload['media_groups'][med]['links']
        if( len(links) == 0 ):
            continue

        city = links[0]['details']['city']
        media = links[0]['details']['name']
        state = links[0]['details']['state']
        single_med_beats = btrp['beat_report'][med] + f' | {state}' + ' | {} | {} | {}'.format( abbreviate_str(city, kwargs['max_city_len']), abbreviate_str(media, kwargs['max_media_len']), abbreviate_str(med, kwargs['max_media_len']) )
        
        print(' {:>5}'.format(i+1), single_med_beats)
    print(' {:>5}'.format(' '), '-'*len(btrp['header_template']))

    btrp['beat_color_maps'] = {y:x for x,y in btrp['beat_color_maps'].items()}
    btrp_len = len(btrp['beat_color_maps'])
    footer = [ get_color_txt(btrp['beat_color_maps'][c], ansi_code=c) for c in kwargs['colors'] if c in btrp['beat_color_maps'] ]
    footer = ' '.join(footer)

    print('Key:')
    print(f'\nTop {btrp_len} beats:', footer)
    if( kwargs['beats_resolver'] != '' ):
        print( '\033[4mUnderline\033[0m: beats resolved with {} method'.format(kwargs['beats_resolver']) )
    print()

    return media_payload

