from iartisanxl.app.event_bus import EventBus
from iartisanxl.modules.common.panels.base_panel import BasePanel


class IPAdapterPanel(BasePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = EventBus()
        self.event_bus.subscribe("ip-adapters", self.on_ip_adapters)
        self.controlnets = []

        self.init_ui()
        self.update_ui()

    def init_ui(self):
        pass

    def update_ui(self):
        pass

    def on_ip_adapters(self, data):
        pass
