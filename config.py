# configuration file, set runtime options here
debug_flag = True

# database connection info
# FIXME: move out to JSON or something, this is retarded and unsafe
db_behaviour = {
    'db_name':     'zlavadna',
    'db_schema':   'behaviour',
    'db_host':     'localhost',
    'db_port':      5432,
    'db_username': 'michalfasanek',
    'db_password': 'Silkroad69'
}

db_metadata = {
    'db_name':     'metadata',
    'db_schema':   'metadata',
    'db_host':     'localhost',
    'db_port':      5432,
    'db_username': 'michalfasanek',
    'db_password': 'Silkroad69'
}


datetime_format = '%Y-%m-%d %H:%M:%S.%f'
dir_logging = 'logging'
dir_img = 'img'
boxplot_png_filename = 'outlier_boxplot.png'

pickle_wrap_amount = 100
session_cutout_time = 1800

unique_partners_ids_in_partners = 'partners_in_partneraddresses.set'
pickle_baseline_filename_partners = 'pickled_partners.dict'
dir_partners = 'partners'

pickle_baseline_filename_users = 'pickled_users_'
dir_users = 'users'

unique_partners_ids_in_deals = 'unique_partners_in_deals_ids.list'
targeted_deals_ids_filename = 'targeted_deals_ids.set'
pickle_baseline_filename_deals = 'pickled_deals.dict'
pickle_deal_max_filename = 'deals_max_amounts'
dir_deals = 'deals'

dir_category = 'categories'
categories_filename = 'categories.dict'

dir_preprocessing = 'preprocessing'
normalization_dict_file = 'normalization_borders.dict'
event_amount_outliers_filename = 'event_amount_outliers.dict'
medians_dict_file = 'event_medians.dict'

dir_vectors = 'vectors'
vectors_baseline_filename = 'vector_tuples_'
dir_partners_grouped_vectors = 'partner_grouped_vectors'
sorted_partner_vectors_baseline_filename = 'sorted_vectors_'
dir_sorted_partner_vectors_input_with_partners = 'partner_grouped_heavy_vectors'
sorted_heavy_partner_vectors_baseline_filename = 'sorted_heavy_vectors_'

user_event_amount_minimum = 20
user_event_split = 10
session_event_amount_minimum = 8

accepted_user_events = [
    'basket_add',
    'view',
    'list',
    'rating',
    'visit_rating_page',
    'purchase_processed'
]

events_targeting_deals = [
    'view',
    'basket_add',
    'rating',
    'visit_rating_page'
]

required_sample_amount = 50
target_sample_amount = 100
flag_heavy_vectors = True
dataset_split = (0.7, 0.2, 0.1)

learning_rate = 0.001
alpha = 0.7
n_hidden_layers = 2
n_hidden_cells_in_layer = 1024
epoch_amount = 25

