import attr


@attr.s
class LoraTrainArgs:
    output_dir = attr.ib()
    model_path = attr.ib()
    rank = attr.ib()
    learning_rate = attr.ib()
    dataset_path = attr.ib()
    batch_size = attr.ib()
    workers = attr.ib()
    accumulation_steps = attr.ib()
    epochs = attr.ib()
    save_steps = attr.ib()
