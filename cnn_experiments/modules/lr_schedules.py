def step_schedule(epoch, lr):
    """
    Learning rate schedule function. Given an epoch index and the associated
    learning rate value, returns an updated value that is decreased by a
    factor `gamma` every 15 epochs.

    Parameters
    ----------
    epoch : int
        Epoch index, starting from 0.

    lr : float
        Current learning rate.
    """
    gamma = 0.1  # Learning rate decrease factor.

    return lr * (gamma ** (epoch // 15))
