from .reshape import Reshape


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)
