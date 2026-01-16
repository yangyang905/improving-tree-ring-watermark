from abc import ABC, abstractmethod

class BaseSync(ABC):
    @abstractmethod
    def add_sync(self, imgs, return_masks=False):
        pass

    @abstractmethod
    def remove_sync(self, imgs, return_info=False):
        pass

