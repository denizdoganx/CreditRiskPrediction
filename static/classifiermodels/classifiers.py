import pickle


def get_all_trained_models():

    ret_dict = dict()

    with open("static\\trainedmodels\\decision_tree_classifier_model.pkl", 'rb') as file:
        decision_tree = pickle.load(file)
        ret_dict["decision_tree"] = decision_tree


    with open("static\\trainedmodels\\knn_classifier_model.pkl", 'rb') as file:
        knn = pickle.load(file)
        ret_dict["knn"] = knn


    with open("static\\trainedmodels\\svm_classifier_model.pkl", 'rb') as file:
        svm = pickle.load(file)
        ret_dict["svm"] = svm
    
    
    return ret_dict