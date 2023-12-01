import attr


@attr.s(eq=True)
class DirectoriesObject:
    models_diffusers = attr.ib(type=str)
    models_safetensors = attr.ib(type=str)
    vaes = attr.ib(type=str)
    models_loras = attr.ib(type=str)
    models_controlnets = attr.ib(type=str)
    models_t2i_adapters = attr.ib(type=str)
    models_ip2_adapters = attr.ib(type=str)
    outputs_images = attr.ib(type=str)
