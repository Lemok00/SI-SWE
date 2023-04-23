import torch

def hiding_process(model, vector, factor):
    with torch.no_grad():
        style = model.mapping(factor)
        container_image = model.generator(vector, style)
    return container_image

def extracting_process(model, container_image, factor, t_max):
    with torch.no_grad():
        recovered_vector = torch.rand(size=(factor.shape[0], model.vector_dim)).to(factor.device) * 2 - 1
        style = model.mapping(factor)
        recovered_image = model.generator(recovered_vector, style)

        for _ in range(t_max):
            difference = model.predictor(recovered_image, container_image)
            recovered_vector += difference
            recovered_vector.clamp(-1, 1)
            recovered_image = model.generator(recovered_vector, style)

        return recovered_vector