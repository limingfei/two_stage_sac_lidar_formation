import re

def select_name(path):

    # 匹配 use_pid_ 后面的 False
    use_pid_match = re.search(r'use_pid_(\w+?)', path)
    names = dict(use_pid=True,speed=0.0,no_lstm=False,leader_mode=None)
    if 'straight' in path:
        names['leader_mode'] = 'straight'
    elif 'circle' in path:
        names['leader_mode'] = 'circle'
    elif 'random' in path:
        names['leader_mode'] = 'random'
    if use_pid_match:
        use_pid_value = use_pid_match.group(1)
        if use_pid_value == 'T':
            names['use_pid'] = True
            names['no_lstm'] = False
        else:
            names['use_pid'] = False
            if 'no_lstm' in path:
                names['no_lstm'] = True
            else:
                names['no_lstm'] = False
    else:
        raise ValueError("use_pid 不存在")

    # 匹配 speed_ 后面的 0.3
    speed_match = re.search(r'speed_([\d.]+)', path)
    if speed_match:
        speed_value = speed_match.group(1)
        names['speed'] = speed_value
    else:
        raise ValueError("speed 不存在")
    return names
