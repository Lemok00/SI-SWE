import torch

def message_to_vector(message, sigma):
    assert sigma == int(sigma)
    secret_vector = torch.zeros(size=(message.shape[0], message.shape[1] // sigma))
    step = 2 / 2 ** sigma
    message_nums = torch.zeros_like(secret_vector)
    for i in range(sigma):
        message_nums += message[:, i::sigma] * 2 ** (sigma - i - 1)
    secret_vector = step * (message_nums + 0.5) - 1
    return secret_vector


def vector_to_message(secret_vector, sigma):
    assert sigma == int(sigma)
    message = torch.zeros(size=(secret_vector.shape[0], secret_vector.shape[1] * sigma))
    step = 2 / 2 ** sigma
    secret_vector = torch.clamp(secret_vector, min=-1, max=1) + 1
    message_nums = secret_vector / step
    zeros = torch.zeros_like(message_nums)
    ones = torch.ones_like(message_nums)
    for i in range(sigma):
        zero_one_map = torch.where(message_nums >= 2 ** (sigma - i - 1), ones, zeros)
        message[:, i::sigma] = zero_one_map
        message_nums -= zero_one_map * 2 ** (sigma - i - 1)
    return message
