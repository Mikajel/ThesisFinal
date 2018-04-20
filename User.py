from datetime import datetime, timedelta
import sql_handler
from typing import List
from Event import Event
import pandas as pd
import config as cfg
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from os import listdir, getcwd, path
import numpy as np
import Category
import Deal


class User(object):
    """
    Defines standard User class.

    User creation should be based on all cookies, to capture non-user cookies and events.
    Create user for every cookie.
    Cookies with no user to pair to will have None attributes for user.
    """

    __slots__ = [
        'cookie',
        'remove_flag',
        'user_id',
        'timestamp_creation',
        'events',
        'event_amount',
    ]

    def __init__(self, db_cursor, cookie: str, event_amount_outliers: {}, employee_ids: set, event_medians: {} or None,
                 categories: {}, deals: {}):

        self.cookie = cookie
        self.remove_flag = False

        user_info = sql_handler.fetch_user_info_by_cookie(db_cursor, self.cookie)

        if user_info is not None:
            self.user_id, self.timestamp_creation = user_info

            # delete employees from users
            if self.user_id in employee_ids:
                self.remove_flag = True
                return

        else:
            self.user_id, self.timestamp_creation = None, None

        self.events = self.get_cookie_events(db_cursor, event_amount_outliers, categories, deals)

        # delete outliers from users
        if self.events is None:
            self.remove_flag = True
            return

        # print('got {} events for cookie {}'.format(len(self.events), cookie))

        self.normalize_user_events(event_medians)
        self.event_amount = len(self.events)

        # NOTE: clip user events
        self.events = self.events[:cfg.user_event_amount_minimum]

        if self.event_amount < cfg.user_event_amount_minimum:
            self.remove_flag = True

    def __str__(self):

        output = ''
        output += '{}{}\n'.format('User ID:'.ljust(20, ' '), self.user_id)
        output += '{}{}\n'.format('Creation time:'.ljust(20, ' '), self.timestamp_creation)

        output += f'Amount of events: {len(self.events)}\n'

        for index, event in enumerate(self.events):
            output += f'Event number: {index}\n'
            output += f'{str(event)}\n\n'

        output += '\n'

        return output

    def get_cookie_events(self, db_cursor, outlier_limits: {}, categories: {}, deals: {}) -> List[Event] or None:
        """
        Get event objects paired to the User object cookie.
        Returns none if User has outlier amount of events

        :return:
        list of Event objects or None
        """

        user_events = sql_handler.fetch_events_by_cookie(db_cursor, self.cookie)

        if not (outlier_limits['low'] < len(user_events) < outlier_limits['high']):
            return None

        event_objects = []
        unique_deal_ids = set(deals.keys())

        for row in user_events:

            event = Event(*row, categories, deals, unique_deal_ids)
            if not event.remove_flag:
                event_objects.append(
                    event
                )

        return event_objects

    def normalize_user_events(self, event_medians: {} or None, cutout_time: int=cfg.session_cutout_time):
        """
        Fills in event durations as difference between events.
        Durations surpassing cutout time are considered end of session.
        These durations(as well as last event) are filled in from event_type medians.
        All events are fetched, after counting durations, uninteresting events are dropped.

        :param event_medians:
        pre-computed event medians to fill in session end events and last user event time

        :param cutout_time:
        time at which session is considered closed, default 1800sec(30min)
        """

        selected_events = []

        # count duration of events and create sessions
        for index, event in enumerate(self.events):

            try:
                event_duration = int(
                    (self.events[index + 1].event_timestamp - self.events[index].event_timestamp).total_seconds()
                )

                if event_duration > cutout_time:
                    event.duration = None
                else:
                    event.duration = event_duration

            # add last user event
            except IndexError:
                    event.duration = None

        for event in self.events:
            event.target_entity_id = int(event.target_entity_id)

        if event_medians is not None:
            self.fill_missing_user_event_durations(event_medians)



    def fill_missing_user_event_durations(self, event_type_medians: {}):

        for event in self.events:
            if event.duration is None:
                event.duration = event_type_medians[event.event_type]


def create_and_save_all_users(cookie_limit: int or None=None):
    """
    Creates a user objects and saves them into a pickle files by a certain amount
    Filters users with outlier amount of events or employee ids
    """

    unique_cookies = sql_handler.fetch_unique_user_cookies(limit=cookie_limit)
    employee_ids = sql_handler.fetch_employee_ids()

    categories = Category.Category.load_categories()
    deals = Deal.load_all_deals()

    print('Gathered {} unique cookies representing users'.format(len(unique_cookies)))
    all_users = []
    pickle_file_counter = 1

    # load pre-processed data
    event_amount_outliers = load_outlier_borders()
    event_duration_medians = load_event_medians()

    # create new users
    with sql_handler.BehaviourDatabaseCursor() as db_cursor:
        for cookie in unique_cookies:

            new_user = User(db_cursor, cookie, event_amount_outliers,
                            employee_ids, event_duration_medians, categories, deals)

            # filtering users
            if new_user.remove_flag:
                continue

            all_users.append(new_user)

            if len(all_users) == cfg.pickle_wrap_amount:

                save_range_start = (pickle_file_counter-1)*cfg.pickle_wrap_amount
                save_range_end = pickle_file_counter * cfg.pickle_wrap_amount
                print(f'Saving users {save_range_start}-{save_range_end}')

                pickle_filename = cfg.pickle_baseline_filename_users + str(pickle_file_counter)

                with open(path.join(getcwd(), cfg.dir_users, pickle_filename), 'wb') as pickle_file:
                    pickle.dump(
                        obj=all_users,
                        file=pickle_file
                    )

                all_users = []
                pickle_file_counter += 1

        # save last < pickle_amount users
        pickle_filename = cfg.pickle_baseline_filename_users + str(pickle_file_counter)

        print(f'Saving last {len(all_users)} users')

        print(all_users[0])

        with open(path.join(getcwd(), cfg.dir_users, pickle_filename), 'wb') as pickle_file:
            pickle.dump(
                obj=all_users,
                file=pickle_file
            )


def count_and_save_event_medians(cookie_limit: int, categories, deals):

    unique_cookies = sql_handler.fetch_unique_user_cookies(limit=cookie_limit)
    employee_ids = sql_handler.fetch_employee_ids()
    event_amount_outliers = load_outlier_borders()

    print(f'Gathered {len(unique_cookies)} unique cookies representing users')
    all_users = []

    unique_event_types = sql_handler.fetch_unique_event_types()
    throwaway_user_amount = 0

    # create new users
    with sql_handler.BehaviourDatabaseCursor() as db_cursor:
        for cookie in unique_cookies:

            new_user = User(db_cursor, cookie, event_amount_outliers, employee_ids, None, categories, deals)

            # filtering users
            if new_user.remove_flag:
                throwaway_user_amount += 1
                continue

            all_users.append(new_user)

    print(f'Throwing away {throwaway_user_amount} users')

    # initialise event type durations dictionary
    event_type_durations = {}
    for event_type in unique_event_types:
        event_type_durations[event_type] = []

    # collect durations
    for user in all_users:
        for event in user.events:
            if event.duration is not None:
                event_type_durations[event.event_type].append(event.duration)

    # get event type medians
    event_type_medians = {}
    for key, value in event_type_durations.items():
        sorted(value)
        if len(value):
            event_type_medians[key] = value[int(
                len(value) / 2
            )]
        else:
            event_type_medians[key] = 1

    for event_type in unique_event_types:
        if event_type not in cfg.accepted_user_events:
            event_type_medians.pop(event_type)

    for event_type, median in event_type_medians.items():
        print(f'Event type:  {event_type}')
        print(f'From amount: {len(event_type_durations[event_type])}')
        print(f'Median:      {median}\n')

    filepath = path.join(getcwd(), cfg.dir_preprocessing, cfg.medians_dict_file)

    with open(filepath, 'wb') as medians_file:

        pickle.dump(event_type_medians, medians_file)


def load_event_medians(debug: bool=False) -> {}:
    """
    Load pre-computed median durations per event type from directory specified in config.

    :return:
    dictionary of median event type durations
    """

    filepath = path.join(getcwd(), cfg.dir_preprocessing, cfg.medians_dict_file)

    with open(filepath, 'rb') as medians_file:
        medians = pickle.load(medians_file)

    if debug:
        for event_type, median in medians.items():
            print(f'Event type:  {event_type}')
            print(f'Median:      {median}\n')

    return medians


def count_and_save_outlier_borders():
    """
    Fetch amounts of events per cookie from the database and count outlier borders.
    Pickle borders as a dict into a file specified in config.
    Save boxplot of outliers into a file specified in config and print it.
    """

    with sql_handler.BehaviourDatabaseCursor() as db_cursor:
        outlier_low, outlier_high = sql_handler.get_ranges_event_cookie_amount(db_cursor)

    # count borders for outliers
    outlier_borders = {
        'low': outlier_low,
        'high': outlier_high
    }
    print(outlier_borders)

    filepath = path.join(getcwd(), cfg.dir_preprocessing, cfg.event_amount_outliers_filename)

    with open(filepath, 'wb') as outlier_file:

        pickle.dump(outlier_borders, outlier_file)


def load_outlier_borders() -> {}:
    """
    Load pre-computed outliers of user event amounts from directory specified in config.

    :return:
    dictionary of outlier borders(keys: ['low', 'high'])
    """

    filepath = path.join(getcwd(), cfg.dir_preprocessing, cfg.event_amount_outliers_filename)

    with open(filepath, 'rb') as outlier_border_file:
        return pickle.load(outlier_border_file)


def show_boxplot(data: [], outlier_borders: {}, save_to_file: bool = False):
    """
    Display a boxplot of outliers.

    :param data:
    data to create a boxplot from

    :param outlier_borders:
    counted outlier borders as dictionary(keys: low, high)
    used to clip boxplot x-axis to the double of the top whisker

    :param save_to_file:
    bool, saving boxplot as png into file specified in 'config.py'
    """

    mpl.use('agg')
    numpy_data = np.array(data)

    bp_figure = plt.figure(1, figsize=(6, 12))
    axes = bp_figure.add_subplot(111)

    axes.set_ylim(0, 2*outlier_borders['high'])

    boxplot = axes.boxplot(numpy_data, sym='+', vert=True, whis=1.5)

    if save_to_file:
        bp_figure.savefig(
            path.join(
                getcwd(),
                cfg.dir_img,
                cfg.boxplot_png_filename
            ),
            bbox_inches='tight'
        )

    bp_figure.show()
    
def get_input_partner_ids(users: [User], deals: {}, partners: {}) -> ([], int):
    """
    Has to pre-fetch input vector partners and find unique set of ids to make target more sparse
        and allow every class to have at least one example for class balancing.

    :param users:
    :param deals:
    :param partners:
    :return:
    """

    input_partners = set()
    users_with_found_partners = 0

    for user in users:

        user_input_partners = []

        # events to construct target vector from
        for event in user.events[:cfg.user_event_split]:

            if event.event_type in cfg.events_targeting_deals:

                id_deal = event.target_entity_id
                deal = deals[id_deal]
                id_partner = deal.id_partner

                if id_partner is not None:
                    user_input_partners.append(id_partner)
        
        if len(user_input_partners):
            input_partners = input_partners.union(set(user_input_partners))
            users_with_found_partners += 1


    return list(input_partners), users_with_found_partners


def get_target_partner_ids(users: [User], deals: {}, partners: {}) -> ([], int):
    """
    Has to pre-fetch target vector partners and find unique set of ids to make target more sparse
        and allow every class to have at least one example for class balancing.

    :param users:
    :param deals:
    :param partners:
    :return:
    """

    target_partners = set()
    users_with_found_partners = 0

    for user in users:

        user_target_partners = []

        # events to construct target vector from
        for event in user.events[cfg.user_event_split:]:

            if event.event_type in cfg.events_targeting_deals:

                id_deal = event.target_entity_id
                deal = deals[id_deal]
                id_partner = deal.id_partner

                if id_partner is not None:
                    user_target_partners.append(id_partner)

        try:
            target_partners.add(user_target_partners[0])
            users_with_found_partners += 1
        except IndexError:
            pass

    return list(target_partners), users_with_found_partners

