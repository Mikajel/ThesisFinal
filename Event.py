from datetime import datetime
import config as cfg


class Event(object):
    """
    On-page user event.

    """

    __slots__ = [
        'unique_hash',
        'event_type',
        'event_timestamp',
        'entity_type',
        'properties',
        'metadata',
        'target_entity_id',
        'target_entity_type',
        'duration',
        'categories',
        'remove_flag'
    ]

    def __init__(self, unique_hash: str, entity_id: int or None, entity_type: str, event_type: str,
                 event_timestamp: str, properties: {}, metadata: {}, target_entity_id: str, target_entity_type: str,
                 categories: {}, deals: {}, unique_deal_ids: set):

        self.unique_hash = unique_hash
        self.event_type = event_type

        if event_type not in cfg.accepted_user_events:
            self.remove_flag = True
            return
        else:
            self.remove_flag = False

        self.unique_hash = unique_hash
        self.event_type = event_type

        self.event_timestamp = event_timestamp

        self.entity_type = entity_type

        self.properties = properties
        self.metadata = metadata

        self.target_entity_type = target_entity_type
        self.duration = None

        if target_entity_type == 'deal':
            if int(target_entity_id) not in unique_deal_ids:
                self.remove_flag = True
                return

        try:
            self.target_entity_id = int(target_entity_id)
        except (TypeError, ValueError):
            self.target_entity_id = None
            self.remove_flag = True
            return

        if event_type != 'purchase_processed':
            self.categories = self.get_event_categories(categories, deals)
        else:
            self.categories = []

    def __str__(self):

        output = ''
        output += 'Timestamp: '.ljust(20, ' ') + str(self.event_timestamp) + '\n'
        output += 'Duration: '.ljust(20, ' ') + str(self.duration) + '\n'
        output += 'Entity type: '.ljust(20, ' ') + str(self.entity_type) + '\n'
        output += 'Event type: '.ljust(20, ' ') + str(self.event_type) + '\n'
        output += 'Categories: '.ljust(20, ' ') + str(self.categories) + '\n'
        output += 'Target entity type: '.ljust(20, ' ') + str(self.target_entity_type) + '\n'
        output += 'Target entity id: '.ljust(20, ' ') + str(self.target_entity_id) + '\n\n'

        return output

    def get_event_categories(self, categories: {}, deals: {}):

        event_category_set = set()

        if self.event_type == 'list':
            return categories[int(self.target_entity_id)].get_category_path(categories)

        try:
            deal_category_ids = deals[int(self.target_entity_id)].categories
        except (KeyError, TypeError):
            print(self)
            raise KeyError
        [event_category_set.add(category_id) for category_id in deal_category_ids]

        # if self.event_type == 'list':
        #     url_string = self.target_entity_id
        #     urls = str(url_string).split('/')[0:]

        return list(event_category_set)




