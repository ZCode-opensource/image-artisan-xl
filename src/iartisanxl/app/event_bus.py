class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EventBus(metaclass=Singleton):
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        # Only append the callback if it's not already in the list
        if callback not in self.subscribers[event_type]:
            self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)

    def publish(self, event_type, data_dict=None):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data_dict)

    def get_all_subscribers(self):
        return self.subscribers

    def get_all_events(self):
        return list(self.subscribers.keys())

    def get_subscribers_for_event(self, event_type):
        return self.subscribers.get(event_type, [])

    def get_events_for_subscriber(self, callback):
        return [event for event, callbacks in self.subscribers.items() if callback in callbacks]
