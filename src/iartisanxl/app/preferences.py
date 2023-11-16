import attr


@attr.s(eq=True)
class PreferencesObject:
    intermediate_images = attr.ib(type=bool)
    use_tomes = attr.ib(type=bool)
    sequential_offload = attr.ib(type=bool)
    model_offload = attr.ib(type=bool)
    save_image_metadata = attr.ib(type=bool)
    save_image_control_annotators = attr.ib(type=bool)
    save_image_control_sources = attr.ib(type=bool)
