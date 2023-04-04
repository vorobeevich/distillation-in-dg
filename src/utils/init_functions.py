import types
import typing as tp


def init_object(module: types.ModuleType,
                obj_config: dict[str, tp.Any]) -> tp.Any:
    """Initializes an object from the module module according to the given config.
    Example:
        module: torch.optim
        object_cfg: {
            name: SGD,
            kwargs: {
                    lr: 0.003,
                    momentum: 0.9
            }
        }
        The function will return an object torch.optim.SGS(lr=0.003, momentum=0.9)

    Args:
        module (types.ModuleType): The module from which the object must be initialized.
        obj_config (typing.Any): the configuration of the object (python dict) to be initialized.
            "name" parameter is responsible for the name of the object.
            "kwargs" parameter is responsible for the object arguments that will be passed through the ** operation.
            For example, if you want to create torch.optim.SGS(lr=0.003, momentum=0.9), you should pass obj_config:
                module: torch.optim
                object_cfg: {
                    name: SGD,
                    kwargs: {
                        lr: 0.003,
                        momentum: 0.9
                    }
                }

    Returns:
        typing.Any: Created object.
    """
    return getattr(module, obj_config["name"])(**obj_config["kwargs"])
