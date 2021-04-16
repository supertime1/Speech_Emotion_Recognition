def decay(epoch):
    if epoch < 50:
        return 1e-3
    elif 50 <= epoch < 150:
        return 1e-4
    else:
        return 1e-5