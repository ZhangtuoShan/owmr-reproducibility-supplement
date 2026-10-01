def merge_dicts(a, b, path=None):
    """merges b into a"""
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def rename_policy_weights(weight_dict, new_name):
    new_weight_dict = {}
    for k, v in weight_dict.items():
        new_agent_name = [new_name] + k.split('/')[1:]
        new_agent_name = '/'.join(new_agent_name)
        new_weight_dict[new_agent_name] = v
    return new_weight_dict