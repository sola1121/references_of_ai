import pickle


output_model_file = "../tmp/saved_model.pkl"

def dump_object(model_object):
    with open(output_model_file, "wb") as file:
        pickle.dump(model_object, file)


def load_object(file_path):
    with open(file_path, "r") as file:
        model_object = pickle.load(file)
    return model_object