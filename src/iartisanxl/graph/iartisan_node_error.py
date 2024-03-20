class IArtisanNodeError(Exception):
    def __init__(self, message, node_name):
        super().__init__(message)
        self.node_name = node_name
