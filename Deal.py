import config as cfg
import pickle
from os import path, getcwd, listdir
import sql_handler
import datetime


class Deal(object):

    __slots__ = [
        'id_',
        'id_partner',
        'price_low',
        'price_high',
        'categories',
        'coupons_created',
        'coupons_canceled',
        'unique_pageviews'
    ]

    def __init__(self, id_: int, id_partner: int, db_cursor: sql_handler.MetadataDatabaseCursor):

        self.id_ = id_
        self.id_partner = id_partner

        self.categories = self.get_deal_categories(db_cursor)
        self.price_low, self.price_high = self.get_dealitems_info(db_cursor)
        self.unique_pageviews, self.coupons_created, self.coupons_canceled = self.get_deal_info(db_cursor)

    def __str__(self):

        output = ''
        output += f'Deal ID: {self.id_}\n'
        output += f'Partner ID: {self.id_partner}\n'
        output += f'Categories: {self.categories}\n'
        output += f'Price from: {self.price_low}\n'
        output += f'Price to: {self.price_high}\n'
        output += f'Unique pageviews: {self.unique_pageviews}\n'
        output += f'Coupons created: {self.coupons_created}\n'
        output += f'Coupons canceled: {self.coupons_canceled}\n'

        return output

    def get_dealitems_info(self, db_cursor: sql_handler.MetadataDatabaseCursor) -> (float, float):

        result_rows = sql_handler.fetch_deal_items(db_cursor, self.id_)

        if not len(result_rows):
            print(f'Deal {self.id_} has no deal items.')

        price_high = 0
        price_low = 100000

        for row in result_rows:
            if row[0] > price_high:
                price_high = row[0]

            if row[0] < price_low:
                price_low = row[0]

        return price_low, price_high

    def get_deal_info(self, db_cursor: sql_handler.MetadataDatabaseCursor) -> (int, int, int):
        """
        Get deal metrics info from metadata database.
        :return:
        unique_pageviews, coupons_created, coupons_canceled
        """

        unique_pageviews = sql_handler.fetch_deal_metrics(db_cursor, self.id_)
        coupons_created, coupons_canceled = sql_handler.fetch_deal_income(db_cursor, self.id_)

        if unique_pageviews is None:
            unique_pageviews = 0
        if coupons_created is None:
            coupons_created = 0
        if coupons_canceled is None:
            coupons_canceled = 0

        return unique_pageviews, coupons_created, coupons_canceled

    def get_deal_categories(self, db_cursor: sql_handler.MetadataDatabaseCursor) -> [int]:

        return sql_handler.fetch_deal_categories(db_cursor, self.id_)


def create_and_save_all_deals():
    """
    Get all deals from database table 'events', pre-process them and pickle as files in dir specified in config.
    Also saves unique deal towns and countries as dict hashed under deal_id into separate files specified in config.
    """

    used_deals_ids = load_all_used_deal_ids()
    deals_db_rows = sql_handler.fetch_deals()
    all_deals = []

    with sql_handler.MetadataDatabaseCursor() as db_cursor:
        for row in deals_db_rows:

            # fetch only deals used in events
            if int(row[0]) in used_deals_ids:
                new_deal = Deal(*row, db_cursor)
                all_deals.append(new_deal)

    all_deals_dict = {}
    for deal in all_deals:
        all_deals_dict[deal.id_] = deal

    with open(path.join(getcwd(), cfg.dir_deals, cfg.pickle_baseline_filename_deals), 'wb') as pickle_file:
        pickle.dump(
            obj=all_deals_dict,
            file=pickle_file
        )

    created_deals_ids = set(all_deals_dict.keys())
    print('Deals found in events but not in database: ')
    print(used_deals_ids - created_deals_ids)

    print(f'\nSaving {len(all_deals_dict.keys())} of deals')


def load_all_deals() -> {}:

    pickle_filename = path.join(getcwd(), cfg.dir_deals, cfg.pickle_baseline_filename_deals)

    with open(pickle_filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def save_all_used_deal_ids():

    used_deal_ids = sql_handler.fetch_targeted_deal_ids()

    with open(path.join(getcwd(), cfg.dir_preprocessing, cfg.targeted_deals_ids_filename), 'wb') as pickle_file:
        pickle.dump(used_deal_ids, pickle_file)


def load_all_used_deal_ids() -> set:

    with open(path.join(getcwd(), cfg.dir_preprocessing, cfg.targeted_deals_ids_filename), 'rb') as pickle_file:
        return pickle.load(pickle_file)



