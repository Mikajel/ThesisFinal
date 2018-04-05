import config as cfg
import pickle
from os import path, getcwd
import sql_handler
import datetime
import Deal


class Partner(object):

    __slots__ = [
        'id_',
        'rating_average',
        'rating_count'
    ]

    def __init__(self, id_, rating_average, rating_count):

        self.id_ = int(id_)

        try:
            self.rating_average = float(rating_average)
        except TypeError:
            self.rating_average = 0

        try:
            self.rating_count = int(rating_count)
        except TypeError:
            self.rating_count = 0

    def __str__(self):

        text = ''
        text += f'Partner ID:    {self.id_}\n'
        text += f'Rating average: {self.rating_average}\n'
        text += f'Rating count:   {self.rating_count}\n\n'

        return text


def create_and_save_partners():

    partner_db_rows = sql_handler.fetch_partners()
    partners = []
    partners_dict = {}

    for row in partner_db_rows:
        partners.append(Partner(*row))

    for partner in partners:
        partners_dict[partner.id_] = partner

    print(f'Amount of partners: {len(partners)}')

    for index in range(5):
        print(partners[index])

    pickle_filename = path.join(getcwd(), cfg.dir_partners, cfg.pickle_baseline_filename_partners)

    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(partners_dict, pickle_file)


def load_all_partners() -> {}:

    pickle_filename = path.join(getcwd(), cfg.dir_partners, cfg.pickle_baseline_filename_partners)

    with open(pickle_filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)






