save_path = './pretrained-model/pretrained-nlp.h5'

def save_model(model):
    model_arch_path = save_path + "-arch"
    model_weights_path = save_path + "-wghts"
    model.save(model_arch_path)
    model.save_weights(model_weights_path)
