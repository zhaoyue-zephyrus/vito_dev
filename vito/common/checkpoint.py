import os
import torch.optim as optim


def is_torch_optimizer(obj):
    return isinstance(obj, optim.Optimizer)


def get_last_ckpt(root_dir):
    if not os.path.exists(root_dir): return None
    ckpt_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt'):
                num_iter = int(filename.split('.ckpt')[0].split('_')[-1])
                ckpt_files[num_iter]=os.path.join(dirpath, filename)
    iter_list = list(ckpt_files.keys())
    if len(iter_list) == 0: return None
    max_iter = max(iter_list)
    return ckpt_files[max_iter]


def load_unstrictly(state_dict, model, loaded_keys=[]):
    missing_keys = []
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except:
                # print(f"{name} mismatch: param {name}, shape {param.data.shape}, state_dict shape {state_dict[name].shape}")
                missing_keys.append(name)
        elif name not in loaded_keys:
            missing_keys.append(name)
    return model, missing_keys


def resume_from_ckpt(state_dict, model_optims, load_optimizer=True):
    all_missing_keys = []
    # load weights first
    for k in model_optims:
        if model_optims[k] and (not is_torch_optimizer(model_optims[k])) and k in state_dict:
            model_optims[k], missing_keys = load_unstrictly(state_dict[k], model_optims[k])
            all_missing_keys += missing_keys
        
    if len(all_missing_keys) == 0 and load_optimizer:
        print("Loading optimizer states")
        for k in model_optims: 
            if model_optims[k] and is_torch_optimizer(model_optims[k]) and k in state_dict:
                model_optims[k].load_state_dict(state_dict[k])
    else:
        print(f"missing weights: {all_missing_keys}, do not load optimzer states")
    return model_optims, state_dict["step"]