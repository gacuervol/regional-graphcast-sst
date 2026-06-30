# INIT_LR = 1.0 #B
# DECAY_RATE = 0.95 #C
# DECAY_STEPS = 5 #D
# epoch_number = 2
lr = 3e-07#INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)

new_params = {}
for key1 in list(params.keys()):
    new_params[key1] = {}
    for key2 in list(params[key1].keys()):
        grads[key1][key2]
        params[key1][key2]
        new_params[key1][key2] = params[key1][key2] - lr * grads[key1][key2]
new_params