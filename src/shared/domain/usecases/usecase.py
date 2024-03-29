from abc import abstractmethod, ABC


class Usecase[R](ABC):
    @abstractmethod
    def execute(self, *args, **kwargs) -> R:
        pass
