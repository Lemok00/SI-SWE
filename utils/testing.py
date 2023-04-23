def type_of_keys(key):
    int_list = ['vector_dim', 'factor_dim', 't_max']
    float_list = [
        'inverse_margin', 'content_margin', 'style_margin', 
        'lambda_rec', 'lambda_inv', 'lambda_gan', 'lambda_content', 'lambda_style']
    if key in int_list:
        return int
    elif key in float_list:
        return float
    else:
        return str

def txt2dict(path):
    temp_dict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            key, value = line.split(': ')
            temp_dict[key] = type_of_keys(key)(value)
    return temp_dict
