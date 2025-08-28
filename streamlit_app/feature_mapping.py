"""
Mapping des noms de variables techniques vers des noms compréhensibles
"""

FEATURE_MAPPING = {
    # Variables démographiques
    'CODE_GENDER': 'Genre',
    'CNT_CHILDREN': 'Nombre d\'enfants',
    'DAYS_BIRTH': 'Âge (jours)',
    'DAYS_EMPLOYED': 'Ancienneté professionnelle (jours)',

    # Variables financières
    'AMT_INCOME_TOTAL': 'Revenus annuels totaux',
    'AMT_CREDIT': 'Montant du crédit',
    'AMT_ANNUITY': 'Montant de l\'annuité',
    'AMT_GOODS_PRICE': 'Prix du bien',

    # Variables de propriété
    'FLAG_OWN_CAR': 'Possède une voiture',
    'FLAG_OWN_REALTY': 'Possède un bien immobilier',

    # Variables de contact
    'FLAG_MOBIL': 'Téléphone mobile',
    'FLAG_EMP_PHONE': 'Téléphone professionnel',
    'FLAG_WORK_PHONE': 'Téléphone au travail',
    'FLAG_CONT_MOBILE': 'Contact mobile',
    'FLAG_PHONE': 'Téléphone',
    'FLAG_EMAIL': 'Email',

    # Variables de type
    'NAME_CONTRACT_TYPE': 'Type de contrat',
    'NAME_TYPE_SUITE': 'Type d\'accompagnement',
    'NAME_INCOME_TYPE': 'Type de revenus',
    'NAME_EDUCATION_TYPE': 'Niveau d\'éducation',
    'NAME_FAMILY_STATUS': 'Situation familiale',
    'NAME_HOUSING_TYPE': 'Type de logement',
    'OCCUPATION_TYPE': 'Type d\'occupation',

    # Variables de région
    'REGION_POPULATION_RELATIVE': 'Population relative de la région',
    'REGION_RATING_CLIENT': 'Note de la région client',
    'REGION_RATING_CLIENT_W_CITY': 'Note de la région client avec ville',
    'REG_CITY_NOT_LIVE_CITY': 'Région différente de la ville de résidence',
    'REG_CITY_NOT_WORK_CITY': 'Région différente de la ville de travail',
    'REG_REGION_NOT_LIVE_REGION': 'Région différente de la région de résidence',
    'REG_REGION_NOT_WORK_REGION': 'Région différente de la région de travail',

    # Variables d'organisation
    'ORGANIZATION_TYPE': 'Type d\'organisation',

    # Variables de documents
    'FLAG_DOCUMENT_2': 'Document 2',
    'FLAG_DOCUMENT_3': 'Document 3',
    'FLAG_DOCUMENT_4': 'Document 4',
    'FLAG_DOCUMENT_5': 'Document 5',
    'FLAG_DOCUMENT_6': 'Document 6',
    'FLAG_DOCUMENT_7': 'Document 7',
    'FLAG_DOCUMENT_8': 'Document 8',
    'FLAG_DOCUMENT_9': 'Document 9',
    'FLAG_DOCUMENT_10': 'Document 10',
    'FLAG_DOCUMENT_11': 'Document 11',
    'FLAG_DOCUMENT_12': 'Document 12',
    'FLAG_DOCUMENT_13': 'Document 13',
    'FLAG_DOCUMENT_14': 'Document 14',
    'FLAG_DOCUMENT_15': 'Document 15',
    'FLAG_DOCUMENT_16': 'Document 16',
    'FLAG_DOCUMENT_17': 'Document 17',
    'FLAG_DOCUMENT_18': 'Document 18',
    'FLAG_DOCUMENT_19': 'Document 19',
    'FLAG_DOCUMENT_20': 'Document 20',
    'FLAG_DOCUMENT_21': 'Document 21',

    # Variables externes
    'EXT_SOURCE_1': 'Score externe 1',
    'EXT_SOURCE_2': 'Score externe 2',
    'EXT_SOURCE_3': 'Score externe 3',

    # Features calculées
    'AGE_YEARS': 'Âge en années',
    'AGE_GROUP': 'Groupe d\'âge',
    'CREDIT_INCOME_RATIO': 'Ratio crédit/revenus',
    'ANNUITY_INCOME_RATIO': 'Ratio annuité/revenus',
    'CREDIT_GOODS_RATIO': 'Ratio crédit/prix du bien',
    'ANNUITY_CREDIT_RATIO': 'Ratio annuité/crédit',
    'AGE_EMPLOYMENT_RATIO': 'Ratio âge/ancienneté',
    'AMT_ANNUITY_MISSING': 'Annuité manquante',
    'CONTACT_SCORE': 'Score de contact',
    'EXT_SOURCES_MEAN': 'Moyenne des scores externes',
    'EXT_SOURCES_MAX': 'Score externe maximum',
    'EXT_SOURCES_MIN': 'Score externe minimum',
    'EXT_SOURCES_STD': 'Écart-type des scores externes',
    'EXT_SOURCES_COUNT': 'Nombre de scores externes',
    'AGE_EXT_SOURCES_INTERACTION': 'Interaction âge/scores externes',

    # Variables temporelles
    'DAYS_REGISTRATION': 'Jours depuis l\'inscription',
    'DAYS_ID_PUBLISH': 'Jours depuis publication ID',
    'DAYS_LAST_PHONE_CHANGE': 'Jours depuis dernier changement téléphone',

    # Variables de famille
    'CNT_FAM_MEMBERS': 'Nombre de membres de la famille',

    # Variables de logement
    'APARTMENTS_AVG': 'Moyenne appartements',
    'BASEMENTAREA_AVG': 'Moyenne surface sous-sol',
    'YEARS_BEGINEXPLUATATION_AVG': 'Moyenne années début exploitation',
    'YEARS_BUILD_AVG': 'Moyenne années construction',
    'COMMONAREA_AVG': 'Moyenne surface commune',
    'ELEVATORS_AVG': 'Moyenne ascenseurs',
    'ENTRANCES_AVG': 'Moyenne entrées',
    'FLOORSMAX_AVG': 'Moyenne étages maximum',
    'FLOORSMIN_AVG': 'Moyenne étages minimum',
    'LANDAREA_AVG': 'Moyenne surface terrain',
    'LIVINGAPARTMENTS_AVG': 'Moyenne appartements habités',
    'LIVINGAREA_AVG': 'Moyenne surface habitable',
    'NONLIVINGAPARTMENTS_AVG': 'Moyenne appartements non habités',
    'NONLIVINGAREA_AVG': 'Moyenne surface non habitable',
    'APARTMENTS_MODE': 'Mode appartements',
    'BASEMENTAREA_MODE': 'Mode surface sous-sol',
    'YEARS_BEGINEXPLUATATION_MODE': 'Mode années début exploitation',
    'YEARS_BUILD_MODE': 'Mode années construction',
    'COMMONAREA_MODE': 'Mode surface commune',
    'ELEVATORS_MODE': 'Mode ascenseurs',
    'ENTRANCES_MODE': 'Mode entrées',
    'FLOORSMAX_MODE': 'Mode étages maximum',
    'FLOORSMIN_MODE': 'Mode étages minimum',
    'LANDAREA_MODE': 'Mode surface terrain',
    'LIVINGAPARTMENTS_MODE': 'Mode appartements habités',
    'LIVINGAREA_MODE': 'Mode surface habitable',
    'NONLIVINGAPARTMENTS_MODE': 'Mode appartements non habités',
    'NONLIVINGAREA_MODE': 'Mode surface non habitable',
    'APARTMENTS_MEDI': 'Médiane appartements',
    'BASEMENTAREA_MEDI': 'Médiane surface sous-sol',
    'YEARS_BEGINEXPLUATATION_MEDI': 'Médiane années début exploitation',
    'YEARS_BUILD_MEDI': 'Médiane années construction',
    'COMMONAREA_MEDI': 'Médiane surface commune',
    'ELEVATORS_MEDI': 'Médiane ascenseurs',
    'ENTRANCES_MEDI': 'Médiane entrées',
    'FLOORSMAX_MEDI': 'Médiane étages maximum',
    'FLOORSMIN_MEDI': 'Médiane étages minimum',
    'LANDAREA_MEDI': 'Médiane surface terrain',
    'LIVINGAPARTMENTS_MEDI': 'Médiane appartements habités',
    'LIVINGAREA_MEDI': 'Médiane surface habitable',
    'NONLIVINGAPARTMENTS_MEDI': 'Médiane appartements non habités',
    'NONLIVINGAREA_MEDI': 'Médiane surface non habitable',

    # Variables de crédit précédent
    'AMT_REQ_CREDIT_BUREAU_HOUR': 'Demande crédit bureau heure',
    'AMT_REQ_CREDIT_BUREAU_DAY': 'Demande crédit bureau jour',
    'AMT_REQ_CREDIT_BUREAU_WEEK': 'Demande crédit bureau semaine',
    'AMT_REQ_CREDIT_BUREAU_MON': 'Demande crédit bureau mois',
    'AMT_REQ_CREDIT_BUREAU_QRT': 'Demande crédit bureau trimestre',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'Demande crédit bureau année',

    # Variables de paiement
    'SK_DPD': 'Jours de retard',
    'SK_DPD_DEF': 'Jours de retard défaut',

    # Variables de bureau de crédit
    'CNT_CHILDREN': 'Nombre d\'enfants',
    'AMT_INCOME_TOTAL': 'Revenus totaux',
    'AMT_CREDIT': 'Montant crédit',
    'AMT_ANNUITY': 'Montant annuité',
    'AMT_GOODS_PRICE': 'Prix bien',

    # Variables de défaut
    'TARGET': 'Cible (défaut)',

    # Variables d'identification
    'SK_ID_CURR': 'ID Client',
    'SK_ID_BUREAU': 'ID Bureau',
    'SK_ID_PREV': 'ID Précédent'
}

def get_readable_feature_name(feature_name):
    """Retourne le nom lisible d'une feature"""
    return FEATURE_MAPPING.get(feature_name, feature_name)

def get_feature_description(feature_name):
    """Retourne une description détaillée d'une feature"""
    descriptions = {
        'EXT_SOURCES_MEAN': 'Moyenne des scores externes de crédit (plus élevé = meilleur risque)',
        'CREDIT_INCOME_RATIO': 'Ratio entre le montant du crédit et les revenus annuels',
        'AGE_YEARS': 'Âge du client en années',
        'DAYS_EMPLOYED': 'Ancienneté professionnelle en jours (négatif = nombre de jours)',
        'AMT_CREDIT': 'Montant total du crédit demandé',
        'AMT_INCOME_TOTAL': 'Revenus annuels totaux du client',
        'AMT_ANNUITY': 'Montant de l\'annuité mensuelle',
        'CODE_GENDER': 'Genre du client (Homme/Femme)',
        'FLAG_OWN_CAR': 'Le client possède-t-il une voiture',
        'FLAG_OWN_REALTY': 'Le client possède-t-il un bien immobilier',
        'CNT_CHILDREN': 'Nombre d\'enfants du client',
        'NAME_INCOME_TYPE': 'Type de revenus du client',
        'NAME_EDUCATION_TYPE': 'Niveau d\'éducation du client',
        'NAME_FAMILY_STATUS': 'Situation familiale du client',
        'NAME_HOUSING_TYPE': 'Type de logement du client',
        'OCCUPATION_TYPE': 'Type d\'occupation professionnelle',
        'ORGANIZATION_TYPE': 'Type d\'organisation employeur',
        'REGION_RATING_CLIENT': 'Note de la région de résidence du client',
        'REGION_RATING_CLIENT_W_CITY': 'Note de la région incluant la ville',
        'REG_CITY_NOT_LIVE_CITY': 'La région de travail diffère de la ville de résidence',
        'REG_REGION_NOT_LIVE_REGION': 'La région de travail diffère de la région de résidence',
        'REG_REGION_NOT_WORK_REGION': 'La région de résidence diffère de la région de travail',
        'LIVE_REGION_NOT_WORK_REGION': 'La région de résidence diffère de la région de travail',
        'REG_CITY_NOT_WORK_CITY': 'La ville de résidence diffère de la ville de travail',
        'LIVE_CITY_NOT_WORK_CITY': 'La ville de résidence diffère de la ville de travail',
        'EXT_SOURCE_1': 'Score externe de crédit 1 (plus élevé = meilleur risque)',
        'EXT_SOURCE_2': 'Score externe de crédit 2 (plus élevé = meilleur risque)',
        'EXT_SOURCE_3': 'Score externe de crédit 3 (plus élevé = meilleur risque)',
        'DAYS_BIRTH': 'Âge en jours (négatif = nombre de jours depuis la naissance)',
        'DAYS_REGISTRATION': 'Nombre de jours depuis l\'inscription',
        'DAYS_ID_PUBLISH': 'Nombre de jours depuis la publication de l\'ID',
        'DAYS_LAST_PHONE_CHANGE': 'Nombre de jours depuis le dernier changement de téléphone',
        'CNT_FAM_MEMBERS': 'Nombre de membres de la famille',
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'Nombre de demandes de crédit au bureau de crédit (heure)',
        'AMT_REQ_CREDIT_BUREAU_DAY': 'Nombre de demandes de crédit au bureau de crédit (jour)',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'Nombre de demandes de crédit au bureau de crédit (semaine)',
        'AMT_REQ_CREDIT_BUREAU_MON': 'Nombre de demandes de crédit au bureau de crédit (mois)',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'Nombre de demandes de crédit au bureau de crédit (trimestre)',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'Nombre de demandes de crédit au bureau de crédit (année)',
        'SK_DPD': 'Jours de retard sur le paiement',
        'SK_DPD_DEF': 'Jours de retard sur le paiement (défaut)',
        'APARTMENTS_AVG': 'Moyenne du nombre d\'appartements dans l\'immeuble',
        'BASEMENTAREA_AVG': 'Moyenne de la surface du sous-sol',
        'YEARS_BEGINEXPLUATATION_AVG': 'Moyenne des années depuis le début de l\'exploitation',
        'YEARS_BUILD_AVG': 'Moyenne des années de construction',
        'COMMONAREA_AVG': 'Moyenne de la surface commune',
        'ELEVATORS_AVG': 'Moyenne du nombre d\'ascenseurs',
        'ENTRANCES_AVG': 'Moyenne du nombre d\'entrées',
        'FLOORSMAX_AVG': 'Moyenne du nombre maximum d\'étages',
        'FLOORSMIN_AVG': 'Moyenne du nombre minimum d\'étages',
        'LANDAREA_AVG': 'Moyenne de la surface du terrain',
        'LIVINGAPARTMENTS_AVG': 'Moyenne du nombre d\'appartements habités',
        'LIVINGAREA_AVG': 'Moyenne de la surface habitable',
        'NONLIVINGAPARTMENTS_AVG': 'Moyenne du nombre d\'appartements non habités',
        'NONLIVINGAREA_AVG': 'Moyenne de la surface non habitable',
        'APARTMENTS_MODE': 'Mode du nombre d\'appartements dans l\'immeuble',
        'BASEMENTAREA_MODE': 'Mode de la surface du sous-sol',
        'YEARS_BEGINEXPLUATATION_MODE': 'Mode des années depuis le début de l\'exploitation',
        'YEARS_BUILD_MODE': 'Mode des années de construction',
        'COMMONAREA_MODE': 'Mode de la surface commune',
        'ELEVATORS_MODE': 'Mode du nombre d\'ascenseurs',
        'ENTRANCES_MODE': 'Mode du nombre d\'entrées',
        'FLOORSMAX_MODE': 'Mode du nombre maximum d\'étages',
        'FLOORSMIN_MODE': 'Mode du nombre minimum d\'étages',
        'LANDAREA_MODE': 'Mode de la surface du terrain',
        'LIVINGAPARTMENTS_MODE': 'Mode du nombre d\'appartements habités',
        'LIVINGAREA_MODE': 'Mode de la surface habitable',
        'NONLIVINGAPARTMENTS_MODE': 'Mode du nombre d\'appartements non habités',
        'NONLIVINGAREA_MODE': 'Mode de la surface non habitable',
        'APARTMENTS_MEDI': 'Médiane du nombre d\'appartements dans l\'immeuble',
        'BASEMENTAREA_MEDI': 'Médiane de la surface du sous-sol',
        'YEARS_BEGINEXPLUATATION_MEDI': 'Médiane des années depuis le début de l\'exploitation',
        'YEARS_BUILD_MEDI': 'Médiane des années de construction',
        'COMMONAREA_MEDI': 'Médiane de la surface commune',
        'ELEVATORS_MEDI': 'Médiane du nombre d\'ascenseurs',
        'ENTRANCES_MEDI': 'Médiane du nombre d\'entrées',
        'FLOORSMAX_MEDI': 'Médiane du nombre maximum d\'étages',
        'FLOORSMIN_MEDI': 'Médiane du nombre minimum d\'étages',
        'LANDAREA_MEDI': 'Médiane de la surface du terrain',
        'LIVINGAPARTMENTS_MEDI': 'Médiane du nombre d\'appartements habités',
        'LIVINGAREA_MEDI': 'Médiane de la surface habitable',
        'NONLIVINGAPARTMENTS_MEDI': 'Médiane du nombre d\'appartements non habités',
        'NONLIVINGAREA_MEDI': 'Médiane de la surface non habitable'
    }

    return descriptions.get(feature_name, f"Variable {feature_name}")

def get_feature_category(feature_name):
    """Retourne la catégorie d'une feature"""
    categories = {
        'demographic': ['CODE_GENDER', 'CNT_CHILDREN', 'DAYS_BIRTH', 'AGE_YEARS', 'AGE_GROUP'],
        'financial': ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO', 'ANNUITY_CREDIT_RATIO'],
        'employment': ['DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'AGE_EMPLOYMENT_RATIO'],
        'education': ['NAME_EDUCATION_TYPE'],
        'family': ['NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS'],
        'housing': ['NAME_HOUSING_TYPE', 'FLAG_OWN_REALTY', 'FLAG_OWN_CAR'],
        'contact': ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CONTACT_SCORE'],
        'region': ['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION'],
        'organization': ['ORGANIZATION_TYPE'],
        'external_scores': ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCES_MEAN', 'EXT_SOURCES_MAX', 'EXT_SOURCES_MIN', 'EXT_SOURCES_STD', 'EXT_SOURCES_COUNT', 'AGE_EXT_SOURCES_INTERACTION'],
        'documents': [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)],
        'temporal': ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'],
        'credit_bureau': ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'],
        'payment': ['SK_DPD', 'SK_DPD_DEF'],
        'building': ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG'],
        'contract': ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE'],
        'missing': ['AMT_ANNUITY_MISSING']
    }

    for category, features in categories.items():
        if feature_name in features:
            return category

    return 'other'
