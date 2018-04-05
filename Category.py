import config as cfg
import datetime
import sql_handler
from os import path, getcwd
from typing import List
import pickle
import unidecode


class Category(object):

    __category_search = None

    __slots__ = [
        'name',
        'id_',
        'id_parent',
        'url'
    ]

    def __init__(self, id_: int, name: str, url: str, id_parent: int):

        self.id_ = int(id_)
        self.name = name
        self.url = url
        self.id_parent = int(id_parent)

    def __str__(self):

        return f'Name: {self.name}\nID: {self.id_}\nURL: {self.url}\nID parent: {self.id_parent}\n'

    def get_category_path(self, all_categories: {}):

        category_path = [self.id_]
        current = self

        while current.id_parent != 0:

            current = all_categories[current.id_parent]
            category_path.append(current.id_)

        return category_path

    @staticmethod
    def create_and_save_categories():
        """
        Create a dict of categories, hashing category objects under category.id_
        """

        categories = []
        category_dict = {}
        rows = sql_handler.fetch_categories()

        for row in rows:
            categories.append(Category(*row))

        for category in categories:
            category_dict[category.id_] = category

        for index in range(20):
            print(categories[index])

        with open(path.join(getcwd(), cfg.dir_category, cfg.categories_filename), 'wb') as pickle_file:
            pickle.dump(
                obj=category_dict,
                file=pickle_file
            )


    @staticmethod
    def load_categories():
        """
        Load category tree.
        :return:
        Root Category object
        """

        with open(path.join(getcwd(), cfg.dir_category, cfg.categories_filename), 'rb') as pickle_file:
            return pickle.load(
                file=pickle_file
            )
