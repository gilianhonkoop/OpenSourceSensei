class DictAsAttributes:
    def __init__(self, data_dict):
        self.__dict__["_data_dict"] = data_dict

    def __getattr__(self, key):
        if key in self._data_dict:
            return self._data_dict[key]
        else:
            raise AttributeError(f"'DictAsAttributes' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self._data_dict[key] = value

    def __delattr__(self, key):
        del self._data_dict[key]


# from sample factory: rl_baseline/sample-factory/sample_factory/utils/utils.py
class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)
