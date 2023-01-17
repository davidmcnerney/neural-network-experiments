from pathlib import Path


class FixtureLoader:
    """
    Convenient class to help you load test fixture files (JSON, HTML etc) which are located in your Python
    source tree. Normally you use the example_file argument, passing it the absolute path to the current
    Python file, and it'll return a FixtureLoader suitable for loading files from the same folder.
    """
    def __init__(self, directory: str = None, example_file: str = None):
        if directory is not None:
            self._directory = Path(directory)
        elif example_file is not None:
            self._directory = Path(example_file).resolve().parent
        else:
            raise Exception("Either the directory or the example_file argument is required")

    def load_file(self, filename: str) -> str:
        path_to_file = self.path_to_file(filename)
        with open(path_to_file) as fh:
            return fh.read()

    def path_to_file(self, filename: str) -> Path:
        return self._directory / Path(filename)


fixture_loader = FixtureLoader(example_file=__file__)
