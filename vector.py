import config as cfg
import datetime
from math import pi, sin, cos
from os import path, getcwd
import Category
import Deal
from copy import copy
import pickle
import sql_handler
import numpy as np
from os import listdir, getcwd
from os.path import join


__not_found_deals = set()
__not_found_partners = set()


def get_not_matched():

    global __not_found_deals
    global __not_found_partners

    return __not_found_deals, __not_found_partners


def create_userfile_vectors(users_filename, deals, partners, categories) -> [tuple]:

    file_counter = int(users_filename.split('_')[-1])

    users_filepath = path.join(getcwd(), cfg.dir_users, users_filename)
    vectors_filepath = path.join(getcwd(), cfg.dir_vectors, cfg.vectors_baseline_filename + str(file_counter))

    with open(users_filepath, 'rb') as users_file:
        users = pickle.load(users_file)

    vectors = []

    for user in users:
        vectors.append(create_user_vectors(user, deals, partners, categories))

    print(f'Pickling {len(vectors)} vectors')

    with open(vectors_filepath, 'wb') as pickle_file:
        pickle.dump(vectors, pickle_file)


def load_userfile_vectors(file_counter: int):

    vectors_filepath = path.join(getcwd(), cfg.dir_vectors, cfg.vectors_baseline_filename + str(file_counter))

    with open(vectors_filepath, 'rb'):
        return pickle.load(vectors_filepath)


def create_user_vectors(user, deals, partners, categories) -> (list, list):
    """
    Create vectors for neural network from user.

    :param deals:
    deals, for creating input vectors and targets(partners from deals)
    :param partners:
    :return:
    list of tuples (input, targets) for user
    """

    categories_ids_list = list(categories.keys())
    categories_ids_list.sort()
    partners_ids_list = list(partners.keys())
    partners_ids_list.sort()

    input_vectors = __create_input_vectors(
        user.events[:cfg.user_event_split],
        deals,
        partners,
        categories_ids_list
    )

    target_vector = __create_target_vector(
        user.events[cfg.user_event_split:],
        deals,
        partners_ids_list
    )

    user_vectors = (input_vectors, target_vector)

    return user_vectors


def __create_input_vectors(events, deals_dict: {}, partner_dict: {}, categories_id_list) -> []:

    def get_month_subvector() -> [float]:

        def sin_cos_normalization() -> (float, float):
            """
            Normalizes time by sin/cos metric
            Source: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

            :return:
            tuple of sin_month, cos_month
            """

            total_months = 12 - 1

            sin_normalized_month = sin(2*pi*current_month/total_months)
            cos_normalized_month = cos(2*pi*current_month/total_months)

            return sin_normalized_month, cos_normalized_month

        current_month = int(event.event_timestamp.month) - 1

        sin_month, cos_month = sin_cos_normalization()

        return [sin_month, cos_month]

    def get_daytime_subvector() -> [float]:

        def sin_cos_normalization() -> (float, float):
            """
            Normalizes time by sin/cos metric
            Source: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

            :return:
            tuple of sin_time, cos_time
            """

            total_day_minutes = 24*60 - 1

            sin_normalized_time = sin(2*pi*current_day_minutes/total_day_minutes)
            cos_normalized_time = cos(2*pi*current_day_minutes/total_day_minutes)

            return sin_normalized_time, cos_normalized_time

        hours = int(event.event_timestamp.hour)
        minutes = int(event.event_timestamp.minute)

        current_day_minutes = hours*60 + minutes

        sin_time, cos_time = sin_cos_normalization()

        return [sin_time, cos_time]

    def get_monthday_subvector() -> [float]:

        def sin_cos_normalization() -> (float, float):
            """
            Normalizes month day by sin/cos metric
            Source: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

            :return:
            tuple of sin_monthday, cos_monthday
            """

            total_month_days = 31 - 1

            sin_normalized_monthday = sin(2 * pi * current_day / total_month_days)
            cos_normalized_monthday = cos(2 * pi * current_day / total_month_days)

            return sin_normalized_monthday, cos_normalized_monthday

        current_day = int(event.event_timestamp.day) - 1

        sin_monthday, cos_monthday = sin_cos_normalization()

        return [sin_monthday, cos_monthday]

    def get_weekday_subvector() -> [float]:

        def sin_cos_normalization() -> (float, float):
            """
            Normalizes weekday by sin/cos metric
            Source: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

            :return:
            tuple of sin_weekday, cos_weekday
            """

            total_weekdays = 7 - 1

            sin_normalized_weekday = sin(2 * pi * current_weekday / total_weekdays)
            cos_normalized_weekday = cos(2 * pi * current_weekday / total_weekdays)

            return sin_normalized_weekday, cos_normalized_weekday

        current_weekday = event.event_timestamp.weekday()

        sin_weekday, cos_weekday = sin_cos_normalization()

        return [sin_weekday, cos_weekday]

    def get_event_type_subvector() -> [int]:

        event_type_subvector = [0]*len(cfg.accepted_user_events)

        event_type_subvector[cfg.accepted_user_events.index(event.event_type)] = 1

        return event_type_subvector

    def get_category_subvector() -> [int]:

        category_subvector = [0]*len(categories_id_list)

        for category in event.categories:
            category_subvector[categories_id_list.index(category)] = 1

        return category_subvector

    def get_price_subvector() -> [float]:

        if event.event_type in cfg.events_targeting_deals:

            target_deal = deals_dict[event.target_entity_id]
            deal_price_low = target_deal.price_low
            deal_price_high = target_deal.price_high

            normalized_price_low = normalize_value(
                value=deal_price_low,
                min_value=normalization_dict['deal_price'][0],
                max_value=normalization_dict['deal_price'][1]
            )
            normalized_price_high = normalize_value(
                value=deal_price_high,
                min_value=normalization_dict['deal_price'][0],
                max_value=normalization_dict['deal_price'][1]
            )

            return [normalized_price_low, normalized_price_high]

        else:
            return [0, 0]

    def get_deal_coupons_subvector():

        if event.event_type in cfg.events_targeting_deals:

            target_deal = deals_dict[event.target_entity_id]
            coupons_created = target_deal.coupons_created
            coupons_canceled = target_deal.coupons_canceled

            normalized_coupons_created = normalize_value(
                value=coupons_created,
                min_value=normalization_dict['coupons_created'][0],
                max_value=normalization_dict['coupons_created'][1]
            )
            normalized_coupons_canceled = normalize_value(
                value=coupons_canceled,
                min_value=normalization_dict['coupons_canceled'][0],
                max_value=normalization_dict['coupons_canceled'][1]
            )

            return [normalized_coupons_created, normalized_coupons_canceled]

        else:
            return [0, 0]

    def get_deal_pageviews_subvector():

        if event.event_type in cfg.events_targeting_deals:
            target_deal = deals_dict[event.target_entity_id]
            pageviews = target_deal.unique_pageviews

            normalized_unique_pageviews = normalize_value(
                value=pageviews,
                min_value=normalization_dict['unique_pageviews'][0],
                max_value=normalization_dict['unique_pageviews'][1]
            )

            return [normalized_unique_pageviews]

        else:
            return [0]

    def get_partner_avg_rating_subvector() -> [float]:

        if event.event_type in cfg.events_targeting_deals:

            target_deal = deals_dict[event.target_entity_id]
            partner_rating_avg = partner_dict[target_deal.id_partner].rating_average

            normalized_avg_rating = normalize_value(
                value=partner_rating_avg,
                min_value=normalization_dict['partner_rating_avg'][0],
                max_value=normalization_dict['partner_rating_avg'][1]
            )

            return [normalized_avg_rating]

        else:
            return [0]

    def get_partner_rating_count_subvector() -> [float]:

        if event.event_type in cfg.events_targeting_deals:

            target_deal = deals_dict[event.target_entity_id]
            partner_rating_count = partner_dict[target_deal.id_partner].rating_count

            normalized_rating_count = normalize_value(
                value=partner_rating_count,
                min_value=normalization_dict['partner_rating_count'][0],
                max_value=normalization_dict['partner_rating_count'][1]
            )

            return [normalized_rating_count]

        else:
            return [0]

    def get_event_duration_subvector() -> [float]:

        event_duration = event.duration
        normalized_duration = normalize_value(
            value=event_duration,
            min_value=normalization_dict['event_durations'][0],
            max_value=normalization_dict['event_durations'][1]
        )

        return [normalized_duration]

    def create_vector() -> []:

        vector = []

        # time and date info
        vector += get_daytime_subvector()
        vector += get_weekday_subvector()
        vector += get_monthday_subvector()
        vector += get_month_subvector()

        # event info
        vector += get_event_type_subvector()
        vector += get_event_duration_subvector()

        # deal and partner metadata info
        vector += get_partner_avg_rating_subvector()
        vector += get_partner_rating_count_subvector()
        vector += get_price_subvector()
        vector += get_deal_coupons_subvector()
        vector += get_deal_pageviews_subvector()

        # categories
        vector += get_category_subvector()

        return vector

    input_vectors = []

    normalization_dict = load_normalization_values()

    for event in events:
        input_vectors.append(create_vector())

    return input_vectors


def __create_target_vector(events, deals_dict: {}, partners_ids_list: []) -> []:

    def gather_events_partners_ids(event_subset: []):

        events_partners_ids = set()

        for event in event_subset:

            if event.event_type in cfg.events_targeting_deals:
                event_deal = deals_dict[event.target_entity_id]
                event_partner_id = event_deal.id_partner
                events_partners_ids.add(event_partner_id)

        return list(events_partners_ids)

    def create_vector(event_subset: []) -> []:

        vector = copy(baseline_target)

        for partner_id in gather_events_partners_ids(event_subset):
            try:
                vector[partners_ids_list.index(partner_id)] = 1
            except ValueError:
                print(f'partner {partner_id} not found in list of ids.')

        return vector

    baseline_target = [0]*len(partners_ids_list)

    return create_vector(events)


def save_normalization_values():
    """
    Partners and deals information are counted on the database side.
    Counts and saves into file values to be normalized:

    deal prices
    partner avg rating
    partner rating count
    deal unique pageviews
    deal coupons created
    deal coupons canceled
    event durations
    """
    normalization_dict = {}

    with sql_handler.MetadataDatabaseCursor() as db_cursor:

        # deals
        normalization_dict['unique_pageviews'] = sql_handler.get_ranges_unique_pageviews(db_cursor)
        normalization_dict['coupons_created'] = sql_handler.get_ranges_coupons_created(db_cursor)
        normalization_dict['coupons_canceled'] = sql_handler.get_ranges_coupons_canceled(db_cursor)
        normalization_dict['deal_price'] = sql_handler.get_ranges_dealitem_prices(db_cursor)

        # partners
        normalization_dict['partner_rating_avg'] = sql_handler.get_ranges_partner_rating_avg(db_cursor)
        normalization_dict['partner_rating_count'] = sql_handler.get_ranges_partner_rating_count(db_cursor)

    event_durations = []

    for filename in listdir(path.join(getcwd(), cfg.dir_users)):
        with open(join(getcwd(), cfg.dir_users, filename), 'rb') as users_file:

            users = pickle.load(users_file)

            for user in users:
                [event_durations.append(event.duration) for event in user.events]

    event_durations.sort()
    quartile_lower = event_durations[int(len(event_durations) * 0.25)]
    quartile_upper = event_durations[int(len(event_durations) * 0.75)]

    iqr = quartile_upper - quartile_lower

    outlier_lower = quartile_lower - 1.5 * iqr
    outlier_upper = quartile_upper + 1.5 * iqr

    normalization_dict['event_durations'] = (max(0.0, float(outlier_lower)), outlier_upper)

    with open(join(getcwd(), cfg.dir_preprocessing, cfg.normalization_dict_file), 'wb') as pickle_file:
        pickle.dump(normalization_dict, pickle_file)

    for key, value in normalization_dict.items():
        print(f'{key}: {value}')


def load_normalization_values() -> {}:

    with open(join(getcwd(), cfg.dir_preprocessing, cfg.normalization_dict_file), 'rb') as pickle_file:
        return pickle.load(pickle_file)


def normalize_value(*, value: float, min_value: float, max_value: float) -> float:

    return (float(value) - min_value) / (max_value - min_value)

