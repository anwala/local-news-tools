import json
import logging

from NwalaTextUtils.textutils import derefURI
from NwalaTextUtils.textutils import genericErrorInfo

logger = logging.getLogger('us_pulse.us_pulse')

def get_us_loc_zipcodes(states_abbrv=[], regions=[]):

    states = {
        "AL": {"zipcode": "36104", "city": "Montgomery", "state": "Alabama"},
        "AK": {"zipcode": "99801", "city": "Juneau", "state": "Alaska"},
        "AZ": {"zipcode": "85001", "city": "Phoenix", "state": "Arizona"},
        "AR": {"zipcode": "72201", "city": "Little Rock", "state": "Arkansas"},
        "CA": {"zipcode": "95814", "city": "Sacramento", "state": "California"},
        "CO": {"zipcode": "80202", "city": "Denver", "state": "Colorado"},
        "CT": {"zipcode": "06103", "city": "Hartford", "state": "Connecticut"},
        "DE": {"zipcode": "19901", "city": "Dover", "state": "Delaware"},
        "FL": {"zipcode": "32301", "city": "Tallahassee", "state": "Florida"},
        "GA": {"zipcode": "30303", "city": "Atlanta", "state": "Georgia"},
        "HI": {"zipcode": "96813", "city": "Honolulu", "state": "Hawaii"},
        "ID": {"zipcode": "83702", "city": "Boise", "state": "Idaho"},
        "IL": {"zipcode": "62701", "city": "Springfield", "state": "Illinois"},
        "IN": {"zipcode": "46225", "city": "Indianapolis", "state": "Indiana"},
        "IA": {"zipcode": "50309", "city": "Des Moines", "state": "Iowa"},
        "KS": {"zipcode": "66603", "city": "Topeka", "state": "Kansas"},
        "KY": {"zipcode": "40601", "city": "Frankfort", "state": "Kentucky"},
        "LA": {"zipcode": "70802", "city": "Baton Rouge", "state": "Louisiana"},
        "ME": {"zipcode": "04330", "city": "Augusta", "state": "Maine"},
        "MD": {"zipcode": "21401", "city": "Annapolis", "state": "Maryland"},
        "MA": {"zipcode": "02201", "city": "Boston", "state": "Massachusetts"},
        "MI": {"zipcode": "48933", "city": "Lansing", "state": "Michigan"},
        "MN": {"zipcode": "55102", "city": "St. Paul", "state": "Minnesota"},
        "MS": {"zipcode": "39205", "city": "Jackson", "state": "Mississippi"},
        "MO": {"zipcode": "65101", "city": "Jefferson City", "state": "Missouri"},
        "MT": {"zipcode": "59623", "city": "Helena", "state": "Montana"},
        "NE": {"zipcode": "68502", "city": "Lincoln", "state": "Nebraska"},
        "NV": {"zipcode": "89701", "city": "Carson City", "state": "Nevada"},
        "NH": {"zipcode": "03301", "city": "Concord", "state": "New Hampshire"},
        "NJ": {"zipcode": "08608", "city": "Trenton", "state": "New Jersey"},
        "NM": {"zipcode": "87501", "city": "Santa Fe", "state": "New Mexico"},
        "NY": {"zipcode": "12207", "city": "Albany", "state": "New York"},
        "NC": {"zipcode": "27601", "city": "Raleigh", "state": "North Carolina"},
        "ND": {"zipcode": "58501", "city": "Bismarck", "state": "North Dakota"},
        "OH": {"zipcode": "43215", "city": "Columbus", "state": "Ohio"},
        "OK": {"zipcode": "73102", "city": "Oklahoma City", "state": "Oklahoma"},
        "OR": {"zipcode": "97301", "city": "Salem", "state": "Oregon"},
        "PA": {"zipcode": "17101", "city": "Harrisburg", "state": "Pennsylvania"},
        "RI": {"zipcode": "02903", "city": "Providence", "state": "Rhode Island"},
        "SC": {"zipcode": "29217", "city": "Columbia", "state": "South Carolina"},
        "SD": {"zipcode": "57501", "city": "Pierre", "state": "South Dakota"},
        "TN": {"zipcode": "37219", "city": "Nashville", "state": "Tennessee"},
        "TX": {"zipcode": "78701", "city": "Austin", "state": "Texas"},
        "UT": {"zipcode": "84111", "city": "Salt Lake City", "state": "Utah"},
        "VT": {"zipcode": "05602", "city": "Montpelier", "state": "Vermont"},
        "VA": {"zipcode": "23219", "city": "Richmond", "state": "Virginia"},
        "WA": {"zipcode": "98507", "city": "Olympia", "state": "Washington"},
        "WV": {"zipcode": "25301", "city": "Charleston", "state": "West Virginia"},
        "WI": {"zipcode": "53703", "city": "Madison", "state": "Wisconsin"},
        "WY": {"zipcode": "82001", "city": "Cheyenne", "state": "Wyoming"},
        "AS": {"zipcode": "96799", "city": "Pago Pago", "state": "American Samoa"},
        "DC": {"zipcode": "20001", "city": "Washington", "state": "District of Columbia"},
        "FM": {"zipcode": "96941", "city": "Kolonia", "state": "Federated States of Micronesia"},
        "GU": {"zipcode": "96910", "city": "Agana (Hagåtña)", "state": "Guam"},
        "MH": {"zipcode": "96960", "city": "Majuro", "state": "Marshall Islands"},
        "MP": {"zipcode": "96950", "city": "Saipan", "state": "Northern Mariana Islands"},
        "PW": {"zipcode": "96939", "city": "Melekeok", "state": "Palau"},
        "PR": {"zipcode": "00901", "city": "San Juan", "state": "Puerto Rico"},
        "VI": {"zipcode": "00802", "city": "Charlotte Amalie", "state": "Virgin Islands"}
    }

    #metro regions according to: https://www.higheredjobs.com/region/
    regions_metro_regions = {
        'NEW_ENGLAND': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'MID_ATLANTIC': ['DE', 'DC', 'MD', 'NJ', 'NY', 'PA'],
        'EASTERN_CENTRAL': ['KY', 'NC', 'TN', 'VA', 'WV'],
        'SOUTHEAST': ['AL', 'AR', 'FL', 'GA', 'LA', 'MS', 'SC'],
        'MIDWEST': ['IL', 'IN', 'MI', 'OH', 'WI'],
        'HEARTLAND': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
        'SOUTHWEST': ['AZ', 'NM', 'OK', 'TX'],
        'ROCKY_MOUNTAIN': ['CO', 'ID', 'MT', 'NV', 'UT', 'WY'],
        'PACIFIC_COAST': ['CA', 'OR', 'WA'],
        'NON_CONTIGUOUS': ['AK', 'HI'],
        'USA': list(states.keys())
    }
    
    for reg in regions:
        if( reg not in regions_metro_regions ):
            continue
        states_abbrv += regions_metro_regions[reg]


    all_locs = []
    dedup_set = set()
    for loc in states_abbrv:
        
        if( loc in dedup_set ):
            continue
        dedup_set.add(loc)

        
        loc = loc.upper()
        if( loc in states ):
            all_locs.append( states[loc] )
    

    return all_locs

def get_us_local_media(country, zipcode_or_city_lst, src_count=100, off_option=''):
    
    logger.info('\nget_us_local_media():')

    payload = {'locations': []}
    if( country == '' or len(zipcode_or_city_lst) == 0 or src_count < 1 ):
        return payload

    
    off_option = off_option.replace(' ', '%20')
    req_size = len(zipcode_or_city_lst)
    for i in range(req_size):
        
        zipcode_or_city = zipcode_or_city_lst[i]
        req_uri = f'http://www.localmemory.org/api/countries/{country}/{zipcode_or_city}/{src_count}/?off={off_option}'

        logger.info(f'\t{i} of {req_size}: {req_uri}')
        try:
            payload['locations'].append( json.loads(derefURI(req_uri)) )
        except:
            genericErrorInfo()
    
    return payload

def get_us_local_media_for_state_or_region(state_abbrv_or_region_lst, src_count=100, off_option='', **kwargs):
    
    regions = []
    states_abbrv = []
    
    for loc in state_abbrv_or_region_lst:
        
        if( len(loc) > 2 ):
            regions.append(loc)
        else:
            states_abbrv.append(loc)

    
    zipcodes = get_us_loc_zipcodes(states_abbrv=states_abbrv, regions=regions)
    zipcodes = [ z['zipcode'] for z in zipcodes ]
    
    return get_us_local_media('USA', zipcodes, src_count=src_count, off_option=off_option)

def get_non_us_media_for_country(country, city, src_count=100, off_option='', **kwargs):
    return get_us_local_media(country, city, src_count=src_count, off_option=off_option)
    
