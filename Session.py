import config as cfg
from Event import Event


class Session(object):
    """
    User session containing events.
    Session is defined as an event-stream cut-off after 30 min of inactivity.
    """
    __slots__ = [
        'events'
    ]

    def __init__(self, events):

        self.events = events

    def __str__(self):

        output = ''
        output += f'Amount of events: {len(self.events)}\n'

        for index, event in enumerate(self.events):
            output += f'Event number: {index}\n'
            output += f'{str(event)}\n\n'

        return output
