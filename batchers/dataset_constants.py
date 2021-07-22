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

_SURVEY_NAMES_2012_16 = {
 'train': ['benin_2012', 'democratic_republic_of_congo_2013','democratic_republic_of_congo_2014', 'guinea_2012', 'kenya_2014',
              'kenya_2015', 'malawi_2012', 'malawi_2014', 'malawi_2015', 'malawi_2016', 'nigeria_2013',
              'nigeria_2015', 'rwanda_2014', 'rwanda_2015', 'senegal_2012',
              'senegal_2013', 'sierra_leone_2013', 'tanzania_2012', 'tanzania_2015', 'tanzania_2016'],
    'val': ['burkina_faso_2014', 'cote_d_ivoire_2012', 'ghana_2014',
            'ghana_2016', 'lesotho_2014', 'togo_2013', 'togo_2014', 'zambia_2013', 'zambia_2014'],
    'test': ['angola_2015', 'angola_2016', 'ethiopia_2016',
             'mali_2012', 'mali_2013', 'mali_2015', 'uganda_2014', 'uganda_2015', 'zimbabwe_2015'],
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

SURVEY_NAMES = {
    '5country': _SURVEY_NAMES_5country,
    '2012-16': _SURVEY_NAMES_2012_16,
    '2012-16A': _SURVEY_NAMES_2012_16A,
    '2012-16B': _SURVEY_NAMES_2012_16B,
    '2012-16C': _SURVEY_NAMES_2012_16C,
    '2012-16D': _SURVEY_NAMES_2012_16D,
    '2012-16E': _SURVEY_NAMES_2012_16E,
}

SIZES = {
    '2012-16': {'train': 7877, 'val': 2607, 'test': 2445, 'all': 12929},
    '2012-16nl': {'all': 12929},
    '2012-16A': {'train': 7534, 'val': 2770, 'test': 2625, 'all': 12929},
    '2012-16B': {'train': 8014, 'val': 2145, 'test': 2770, 'all': 12929},
    '2012-16C': {'train': 8543, 'val': 2241, 'test': 2145, 'all': 12929},
    '2012-16D': {'train': 7540, 'val': 3148, 'test': 2241, 'all': 12929},
    '2012-16E': {'train': 7156, 'val': 2625, 'test': 3148, 'all': 12929},
    'incountryA': {'train': 7751, 'val': 2589, 'test': 2589, 'all': 12929},  
    'incountryB': {'train': 7751, 'val': 2589, 'test': 2589, 'all': 12929},  
    'incountryC': {'train': 7751, 'val': 2589, 'test': 2589, 'all': 12929},  
    'incountryE': {'train': 7752, 'val': 2588, 'test': 2588, 'all': 12929},  
    'incountryD': {'train': 7752, 'val': 2588, 'test': 2588, 'all': 12929},  
}

URBAN_SIZES = {
    '2012-16': {'train': 2647, 'val': 1031, 'test': 916, 'all': 4594},
    '2012-16A': {'train': 2810, 'val': 847, 'test': 937, 'all': 4594},
    '2012-16B': {'train': 3002, 'val': 745, 'test': 847, 'all': 4594},
    '2012-16C': {'train': 3016, 'val': 833, 'test': 745, 'all': 4594},
    '2012-16D': {'train': 2529, 'val': 1232, 'test': 833, 'all': 4594},
    '2012-16E': {'train': 2425, 'val': 937, 'test': 1232, 'all': 4594},
}
RURAL_SIZES = {
    '2012-16': {'train': 5230, 'val': 1576, 'test': 1529, 'all': 8335},
    '2012-16A': {'train': 4724, 'val': 1923, 'test': 1688, 'all': 12868},
    '2012-16B': {'train': 5012, 'val': 1400, 'test': 1923, 'all': 12868},
    '2012-16C': {'train': 5527, 'val': 1408, 'test': 1400, 'all': 12868},
    '2012-16D': {'train': 5011, 'val': 1916, 'test': 1408, 'all': 12868},
    '2012-16E': {'train': 4731, 'val': 1688, 'test': 1916, 'all': 12868},
}

# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands

_MEANS_2012_16 = {
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
_MEANS_2012_16nl = {
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

_STD_DEVS_2012_16 = {
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
_STD_DEVS_2012_16nl = {
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
    '2012-16': _MEANS_2012_16,
    '2012-16nl': _MEANS_2012_16nl,
}

STD_DEVS_DICT = {
    '2012-16': _STD_DEVS_2012_16,
    '2012-16nl': _STD_DEVS_2012_16nl,
}
