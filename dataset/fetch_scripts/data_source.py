from collections.abc import Callable
from typing import Generator


class DataSource:
    script: Callable[str, str]
    stream: Callable[[str, int], str]
    gen: Generator = None

    def __init__(self, script, stream):
        self.script = script
        self.stream = stream

    def set_stream(self, language, max_size):
        self.gen = self.stream(self.script(language), max_size)

    def __iter__(self):
        if self.gen is None:
            raise RuntimeError(
                "generator is not defined on DataSource. call set_stream first"
            )
        for element in self.gen:
            yield element
