def decay(epoch):
    if epoch < 50:
        return 1e-4
    elif 50 <= epoch < 150:
        return 1e-5
    else:
        return 1e-6
