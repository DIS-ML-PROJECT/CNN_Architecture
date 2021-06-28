DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mali', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

_SURVEY_NAMES_5country = {
    'train': ['uganda_2014', 'tanzania_2012', 'rwanda_2014', 'nigeria_2013'],
    'val': ['malawi_2012'],
    'test': []
}

_SURVEY_NAMES_20012_16 = {
    'train': ['benin_2012', 'democratic_republic_of_congo_2013', 'guinea_2012', 'kenya_2014',
              'kenya_2015', 'malawi_2012', 'malawi_2014', 'malawi_2015', 'malawi_2016',
              'nigeria_2013', 'nigeria_2015', 'rwanda_2014',
              'senegal_2012', 'sierra_leone_2013', 'tanzania_2015',
              'togo_2013', 'uganda_2014', 'zimbabwe_2015'],
    'val': ['burkina_faso_2014', 'cote_d_ivoire_2012', 'ghana_2014',
            'ghana_2016', 'lesotho_20014', 'zambia_2013'],
    'test': ['angola_2015', 'angola_2016', 'ethiopia_2016', 'mali_2012', 
            'mali_2013', 'mali_2015', 'tanzania_2012', 'tanzania_2015'],
}

_SURVEY_NAMES_2012_16A = {
    'train': ['democratic_republic_of_congo', 'ghana', 'kenya',
              'lesotho', 'malawi', 'nigeria', 'senegal',
              'togo', 'uganda', 'zambia', 'zimbabwe'],
    'val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    'test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
}
_SURVEY_NAMES_2012_16B = {
    'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
              'ethiopia', 'kenya', 'lesotho', 'mali',
              'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
    'val': ['ghana', 'malawi', 'zimbabwe'],
    'test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
}
_SURVEY_NAMES_2012_16C = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
              'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
              'sierra_leone', 'tanzania', 'zambia'],
    'val': ['democratic_republic_of_congo', 'nigeria', 'togo', 'uganda'],
    'test': ['ghana', 'malawi', 'zimbabwe'],
}
_SURVEY_NAMES_2012_16D = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire',
              'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
              'sierra_leone', 'tanzania', 'zimbabwe'],
    'val': ['kenya', 'lesotho', 'senegal', 'zambia'],
    'test': ['democratic_republic_of_congo', 'nigeria', 'togo', 'uganda'],
}
_SURVEY_NAMES_2012_16E = {
    'train': ['benin', 'burkina_faso', 'democratic_republic_of_congo',
              'ghana', 'guinea', 'malawi', 'nigeria', 'sierra_leone',
              'tanzania', 'togo', 'uganda', 'zimbabwe'],
    'val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    'test': ['kenya', 'lesotho', 'senegal', 'zambia'],
}

_SURVEY_NAMES_LSMS = ['ethiopia_2011', 'ethiopia_2015', 'malawi_2010', 'malawi_2016',
                      'nigeria_2010', 'nigeria_2015', 'tanzania_2008', 'tanzania_2012',
                      'uganda_2005', 'uganda_2009', 'uganda_2013']

SURVEY_NAMES = {
    '5country': _SURVEY_NAMES_5country,
    '2012-16': _SURVEY_NAMES_2009_17,
    '2012-16A': _SURVEY_NAMES_2009_17A,
    '2012-16B': _SURVEY_NAMES_2009_17B,
    '2012-16C': _SURVEY_NAMES_2009_17C,
    '2012-16D': _SURVEY_NAMES_2009_17D,
    '2012-16E': _SURVEY_NAMES_2009_17E,
    'LSMS': _SURVEY_NAMES_LSMS,
}

SIZES = {
    '2012-16': {'train': 1, 'val': 1, 'test': 1, 'all': 3},
    #'2012-16': {'train': 12319, 'val': 3257, 'test': 4093, 'all': 19669},
    '2012-16nl': {'all': 261396},
    '2012-16A': {'train': 11797, 'val': 3909, 'test': 3963, 'all': 19669},
    '2012-16B': {'train': 11820, 'val': 3940, 'test': 3909, 'all': 19669},
    '2012-16C': {'train': 11800, 'val': 3929, 'test': 3940, 'all': 19669},
    '2012-16D': {'train': 11812, 'val': 3928, 'test': 3929, 'all': 19669},
    '2012-16E': {'train': 11778, 'val': 3963, 'test': 3928, 'all': 19669},
    'incountryA': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'incountryB': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'incountryC': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'incountryD': {'train': 11802, 'val': 3933, 'test': 3934, 'all': 19669},
    'incountryE': {'train': 11802, 'val': 3934, 'test': 3933, 'all': 19669},
}

URBAN_SIZES = {
    '2012-16': {'train': 3954, 'val': 1212, 'test': 1635, 'all': 6801},
    '2012-16A': {'train': 4264, 'val': 1221, 'test': 1316, 'all': 6801},
    '2012-16B': {'train': 4225, 'val': 1355, 'test': 1221, 'all': 6801},
    '2012-16C': {'train': 4010, 'val': 1436, 'test': 1355, 'all': 6801},
    '2012-16D': {'train': 3892, 'val': 1473, 'test': 1436, 'all': 6801},
    '2012-16E': {'train': 4012, 'val': 1316, 'test': 1473, 'all': 6801},
}

RURAL_SIZES = {
    '2012-16': {'train': 8365, 'val': 2045, 'test': 2458, 'all': 12868},
    '2012-16A': {'train': 7533, 'val': 2688, 'test': 2647, 'all': 12868},
    '2012-16B': {'train': 7595, 'val': 2585, 'test': 2688, 'all': 12868},
    '2012-16C': {'train': 7790, 'val': 2493, 'test': 2585, 'all': 12868},
    '2012-16D': {'train': 7920, 'val': 2455, 'test': 2493, 'all': 12868},
    '2012-16E': {'train': 7766, 'val': 2647, 'test': 2455, 'all': 12868},
}

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands

_MEANS_2009_17 = {
    'BLUE':  0.059183,
    'GREEN': 0.088619,
    'RED':   0.104145,
    'SWIR1': 0.246874,
    'SWIR2': 0.168728,
    'TEMP1': 299.078023,
    'NIR':   0.253074,
    'DMSP':  4.005496,
    'VIIRS': 1.096089,
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}
_MEANS_2009_17nl = {
    'BLUE':  0.063927,
    'GREEN': 0.091981,
    'RED':   0.105234,
    'SWIR1': 0.235316,
    'SWIR2': 0.162268,
    'TEMP1': 298.736746,
    'NIR':   0.245430,
    'DMSP':  7.152961, #wealthpooled
    'VIIRS': 2.322687,
}

_STD_DEVS_2009_17 = {
    'BLUE':  0.022926,
    'GREEN': 0.031880,
    'RED':   0.051458,
    'SWIR1': 0.088857,
    'SWIR2': 0.083240,
    'TEMP1': 4.300303,
    'NIR':   0.058973,
    'DMSP':  23.038301,
    'VIIRS': 4.786354,
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}
_STD_DEVS_2009_17nl = {
    'BLUE':  0.023697,
    'GREEN': 0.032474,
    'RED':   0.051421,
    'SWIR1': 0.095830,
    'SWIR2': 0.087522,
    'TEMP1': 6.208949,
    'NIR':   0.071084,
    'DMSP':  29.749457,
    'VIIRS': 14.611589,
}


MEANS_DICT = {
    '2012-16': _MEANS_2009_17,
    '2012-16nl': _MEANS_2009_17nl,
}

STD_DEVS_DICT = {
    '2012-16': _STD_DEVS_2009_17,
    '2012-16nl': _STD_DEVS_2009_17nl,
}

