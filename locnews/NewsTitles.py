import logging
import re

from NwalaTextUtils.textutils import getDedupKeyForURI
from NwalaTextUtils.textutils import getDomain
from NwalaTextUtils.textutils import getLinks
from NwalaTextUtils.textutils import getUriDepth

from urllib.parse import urlparse

from locnews.util import isStopword
from locnews.util import word_tokenizer

logger = logging.getLogger('us_pulse.us_pulse')

class NewsTitles:

    def __init__(self, homepage_url, details, max_links=-1):
        
        self.homepage_url = homepage_url
        self.details = details
        self.max_links = max_links

    @staticmethod
    def get_title_set( title, stopwords, split_pattern="[^a-zA-Z0-9.'’]" ):

        title_set = title.lower().strip()
        title_set = word_tokenizer( title_set, split_pattern=split_pattern )
        title_set = [ w for w in title_set if isStopword(w, stopwords) == False ]

        return title_set

    @staticmethod
    def format_news_beats(possible_beats, details=None):

        beat_stopwords = {
            'jan', 
            'january', 
            'feb', 
            'february',
            'mar',
            'march',
            'apr',
            'april',
            'may',
            'jun',
            'june',
            'jul',
            'july',
            'aug',
            'august',
            'sep',
            'september',
            'oct',
            'october',
            'nov',
            'november',
            'dec',
            'december'
        }

        possible_beats = [ b for b in possible_beats if b not in beat_stopwords ]# 2b (see get_news_beat_candidates())
        possible_beats = [{'is_specific': True, 'beat': b, 'details': details} for b in possible_beats]
        
        return possible_beats

    @staticmethod
    def get_news_beat_candidates( link, entities, stopwords, min_t_wrd_cnt, split_pattern ):

        '''
            beat definition 
            1. remove numeric beat cands
            2.  if beat word count is < min_t_wrd_cnt
                    a. keep
                b. if beat word is month remove
        '''
        

        scheme, netloc, path, params, query, fragment = urlparse(link)
        path_set = NewsTitles.get_title_set( path, stopwords, split_pattern=split_pattern )
        path = path.strip()
        path = path[:-1] if path[-1] == '/' else path
        path = path[1:] if path[0] == '/' else path

        possible_beats = path.lower().split('/')[:-1]
        possible_beats = [ b for b in possible_beats if re.search('[a-zA-Z]', b) is not None ]#def 1.
        
        possible_beats = [ b.split('-') for b in possible_beats ]
        possible_beats = [ '-'.join(b) for b in possible_beats if len(b) < min_t_wrd_cnt ]#def 2a.

        possible_beats = NewsTitles.format_news_beats( possible_beats )

        return possible_beats

    def extract_links(self, stopwords, html='', split_pattern="[^a-zA-Z0-9.'’]", **kwargs):

        '''
            Conditions for filtering links
            1. remove links with blank titles
            2. remove links with titles that don't have spaces
            3. remove links with from different domain
            4. remove links with titles smaller than min_t_wrd_cnt
            5. remove links with depth smaller than min_uri_dpt
        '''

        kwargs.setdefault('rm_out_domain', True)
        kwargs.setdefault('min_t_wrd_cnt', 3)
        kwargs.setdefault('min_uri_dpt', 2)
        kwargs.setdefault('deref_uri', True)
        
        media = []
        min_t_wrd_cnt = kwargs['min_t_wrd_cnt']
        min_uri_dpt = kwargs['min_uri_dpt'] 
        

        filter_report = {
            'title_blank': 0,
            'rm_out_domain': 0,
            'title_no_space': 0,
            f'word_count < {min_t_wrd_cnt}': 0,
            f'depth < {min_uri_dpt}': 0
        }

        if( kwargs['deref_uri'] is False and html == '' ):
            return []


        old_links = getLinks( uri=self.homepage_url, html=html, fromMainTextFlag=False )
        if( len(old_links) == 0 ):
            return []


        links = []
        uri_dedup_set = set()
        for u in old_links:

            uri_ky = getDedupKeyForURI( u['link'] )
            if( uri_ky in uri_dedup_set ):
                continue

            links.append(u)
            uri_dedup_set.add(uri_ky)

        
        domain = getDomain( self.homepage_url )
        for i in range(len(links)):

            links[i]['pos'] = i
            links[i]['depth'] = getUriDepth( links[i]['link'] )
            links[i]['details'] = self.details
            links[i]['details']['domain'] = domain
            links[i]['beat_candidates'] = []

            if( links[i]['title'] == '' ):
                filter_report['title_blank'] += 1
                continue

            if( links[i]['title'].find(' ') == -1 ):
                #filter out "link" titles e.g, "img.full-width1{flex-shrink:0;object-fit:contain;min-height:50%;height:auto;width:100%}"
                filter_report['title_no_space'] += 1
                continue

            title_set = NewsTitles.get_title_set( links[i]['title'], stopwords, split_pattern=split_pattern )
            links[i]['entities'] = [ {'entity': e, 'class': 'title'} for e in title_set ]

            if( links[i]['link'].find(domain) == -1 and kwargs['rm_out_domain'] is True ):
                filter_report['rm_out_domain'] += 1
                continue

            word_count = len( links[i]['entities'] )
            if( word_count < min_t_wrd_cnt ):
                filter_report[f'word_count < {min_t_wrd_cnt}'] += 1
                continue

            if( links[i]['depth'] < min_uri_dpt ):
                filter_report[f'depth < {min_uri_dpt}'] += 1
                continue

            links[i]['beat_candidates'] = NewsTitles.get_news_beat_candidates( links[i]['link'], links[i]['entities'], stopwords, min_t_wrd_cnt, split_pattern )
            media.append( links[i] )

        if( self.max_links != -1 ):
            media = media[:self.max_links]

        if( len(media) == 0 ):
            logger.info( f'\nextract_links(), filter_report for {self.homepage_url}: ' + str(filter_report) )

        return media
