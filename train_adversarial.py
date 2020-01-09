import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from collections import defaultdict
import json 
import math

from toy_dataset import ToyDataset

# You may want to look into tensorboardX for logging
from tensorboardX import SummaryWriter



class Predictor(nn.Module):
    def __init__(self, latent_dim): # latent_dim argument in order to make interpolation.py work 
        super(Predictor, self).__init__()
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

        self.network_G = nn.Sequential(
            nn.Linear(latent_dim, 1))



    def forward(self, z):
        # Generate images from z
        output = self.network_G(z)

        logits = torch.tanh(output)

        return output, logits


class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()



        self.network_D = nn.Sequential(
            nn.Linear(1, 1))

    def forward(self, img):
        # return discriminator score for img

        output = self.network_D(img)
        logits = torch.sigmoid(output)

        return output, logits 







def train(dataloader, adversary, predictor, optimizer_P, optimizer_A):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))


    # load data
    train_dataset = ToyDataset(5)
    val_dataset = ToyDataset(5)
    test_dataset = ToyDataset(5)


    dataloader_train = DataLoader(train_dataset, args.batch_size)
    dataloader_val = DataLoader(val_dataset, args.batch_size)
    dataloader_test = DataLoader(test_dataset, args.batch_size)


    features_dim = dataset_train.x.shape[1]

    # Initialize models 
    predictor = Predictor(features_dim).to(device)
    adversary = Adversary().to(device)

    # initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.lr)


    criterion_P = nn.BCELoss()
    criterion_A = nn.BCELoss()



    step = 1

    for epoch in range(args.n_epochs):

        for i, (x, y, z) in enumerate(dataloader_train):

            x_train = x.to(device)
            true_y_label = y.to(device)
            true_z_label = z.to(device)


            # forward step predictior 
            pred_y_label, pred_y_logit = predictor(x_train) 

            # compute loss predictor 
            loss_P = criterion_P(pred_y_label, true_y_label)



            if args.debias:
                # forward step adverserial 
                pred_z_label, pred_z_logit = adversary(pred_y_logit)

                # compute loss adverserial 
                loss_A = criterion_A(pred_z_label, true_z_label)

                # reset gradients adversary
                optimizer_A.zero_grad()

                # compute gradients adversary
                loss_A.backward(retain_graph=True)

                # update adversary params #### NOTE: maybe move to the end
                optimizer_A.step()

                # concatenate gradients of adversary params 
                grad_w_La  = get_grad(predictor)


            #### UPDATE PREDICTOR ### 

            # reset gradients
            optimizer_P.zero_grad()

            # compute gradients 
            loss_P.backward()

            if args.debias: 

                # concatenate gradients of predictor params 
                grad_w_Lp = get_grad(predictor)

                proj_grad = (torch.dot(grad_w_Lp, grad_w_La) / torch.dot(grad_w_La, grad_w_La)) * grad_w_La



                alpha = math.sqrt(step) 

                grad_w_Lp = grad_w_Lp - proj_grad -alpha * grad_w_La


                replace_grad(predictor, grad_w_Lp)



            optimizer_P.step()




            # run validation 
            if step % args.eval_freq == 0: 

                with torch.no_grad(): 



                    for i, (x, y, z) in enumerate(dataloader_val):

                        x_val = x.to(device)
                        true_y_label = y.to(device)
                        true_z_label = z.to(device)


                        # forward step predictior 
                        pred_y_label, pred_y_logit = predictor(x_train) 

                        # compute loss predictor 
                        loss_P_val = criterion_P(pred_y_label, true_y_label)


                        if args.debias:
                            # forward step adverserial 
                            pred_z_label, pred_z_logit = adversary(pred_y_logit)

                            # compute loss adverserial 
                            loss_A_val = criterion_A(pred_z_label, true_z_label)

            step += 1




def get_grad(model): 

    g = None 

    for name, param in model.named_parameters():

        grad = param.grad
        if "bias" in name: 
            grad = grad.unsqueeze(dim=1)



        if g is None: 
            g = param.grad
        else:
            g = torch.cat((g, grad), dim=1)

    return g.squeeze(dim=0)


def replace_grad(model, grad): 


    start = 0

    for name, param in model.named_parameters(): 

        numel = param.numel()

        param.grad.data = grad[start:start + numel].expand_as(param.grad)


        start += numel

    return





def main():

	#
	# NEEDS TI BE ADJUSTED!! 
	######

    # # Create output image directory
    # os.makedirs('images', exist_ok=True)

    # Start training
    train()



    # # You can save your predictor here to re-use it to generate images for your
    # # report, e.g.:
    # torch.save(predictor.state_dict(), "mnist_predictor.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')

    parser.add_argument('--eval_freq', type=int, default=500,
                        help='Frequency of evaluation on the validation set')

    parser.add_argument('--print_every', type=int, default=100, 
                        help='number of iterations after which the training progress is printed')
    parser.add_argument('--debias', type=bool, default=True, 
                        help='Use the adversial network to mitigate unwanted bias')
    args = parser.parse_args()


    main()