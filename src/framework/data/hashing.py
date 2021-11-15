import hashlib


class Hasher:
    def __init__(self):
        self.__hasher = hashlib.sha3_256()
        self.__encoding = "utf-8"
        return

    def compute_hash(self, data: any) -> str:
        self.__hasher.update(repr(data).encode(self.__encoding))
        return self.__hasher.hexdigest()
