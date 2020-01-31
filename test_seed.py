import torch
import datasets
import os
from utils import plot_adult_results
from sklearn.metrics import accuracy_score, mean_squared_error
from torch import nn
from main import test 
from model import Predictor, Adversary
import seaborn as sns
import matplotlib.pyplot as plt   
import numpy as np

ROOT_DIR = (os.path.abspath(''))
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_results(accuracy, neg_auc, pos_auc, model_type=""):
    print("-------------------------------------------------------------------")
    print("RESULTS %s \n"%(model_type.upper()))
    print("Test accuracy predictor: {:.5f}".format(accuracy))
    print("AUC Female: {:.5f} || AUC Male: {:.5f}".format(neg_auc, pos_auc))
    print("-------------------------------------------------------------------")
    
np.random.seed(40)
torch.manual_seed(40)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

__, __, dataloader_test = datasets.utils.get_dataloaders(batch_size=128, dataset='images')

print(next(iter(dataloader_test))[0])

pred_biased_filename = "pred_debiased_False_images_seed_40"
pred_debiased_filename = "pred_debiased_True_images_seed_40"
input_dim = next(iter(dataloader_test))[0].shape[1]
output_dim = next(iter(dataloader_test))[1].shape[1]

# Load the biased model 
predictor_biased = ImagePredictor(input_dim, output_dim).to(DEVICE)
predictor_biased.load_state_dict(torch.load(os.path.join(MODEL_DIR, pred_biased_filename), map_location=DEVICE))
predictor_biased.eval();

# Load the debiased model
predictor_debiased = ImagePredictor(input_dim, output_dim).to(DEVICE)
predictor_debiased.load_state_dict(torch.load(os.path.join(MODEL_DIR, pred_debiased_filename), map_location=DEVICE))
predictor_debiased.eval();

# Results biased predictor
accuracy_b, neg_auc_b, pos_auc_b = test(dataloader_test, 
                   predictor = predictor_biased, 
                   adversary = None, 
                   criterion = nn.BCELoss(), 
                   metric = accuracy_score, 
                   device = DEVICE, 
                   dataset_name = 'images', 
                   show_logs = False)
                                                                                                        

# Results debiased predictor 
accuracy_db, neg_auc_db, pos_auc_db = test(dataloader_test, 
                   predictor = predictor_debiased, 
                   adversary = None, 
                   criterion = nn.BCELoss(), 
                   metric = accuracy_score, 
                   device = DEVICE, 
                   dataset_name = 'images', 
                   show_logs = False)


print_results(accuracy_b, neg_auc_b, pos_auc_b, model_type="Biased")
print_results(accuracy_db, neg_auc_db, pos_auc_db, model_type="Debiased")