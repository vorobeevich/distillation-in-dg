def init_object(module, obj_config):
    return getattr(module, obj_config["name"])(**obj_config["kwargs"])

def init_object_list(module, config_list):
    res = []
    for obj in config_list:
        res.append(init_object(module, obj))
    return res
