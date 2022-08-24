from model import TCN
import pandas as pd
from utils import BreatheDataset
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from torch.utils.data import Dataset,DataLoader
from plotgraph import draw_compare_result

class SimulationDataset(Dataset):
    def __init__(self,dataset):
        self.database, self.GT_database = dataset["X"], dataset["Y"]

    def __getitem__(self,index):
        data = self.database[index]
        data = torch.from_numpy(data) #DoubleTensor
        GT = self.GT_database[index]
        GT = torch.from_numpy(GT) #DoubleTensor
        return data,GT

    def __len__(self):
        return len(self.database)

def run(dataset, name='run'):
    model.eval()
    total_batch_loss = 0.0 #to save the sum of all batch loss
    count = 0 # count how many batch
    model_prediction_list = []
    ground_truth_list = []
    total_time_cost = 0.0

    with torch.no_grad():
        for i,(one_batch_data) in enumerate(dataset):
            x, y = one_batch_data
            x, y = Variable(x), Variable(y)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            start = time.time()
            output = model(x)
            end = time.time()
            time_cost_for_one_run = end - start
            # print('time cost', time_cost_for_one_run)
            total_time_cost += time_cost_for_one_run
            model_prediction_list.append(output[0][-1].item())
            ground_truth_list.append(y[0][-1].item())
            criterion = nn.MSELoss(reduction='mean') #####
            loss = criterion(output,y)
            total_batch_loss += loss.item()
            count += 1
        eval_loss = total_batch_loss/count  # epoch loss
        avg_time_cost = total_time_cost/count
        print("avg time cost : " ,avg_time_cost)
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss, model_prediction_list, ground_truth_list

if __name__ == "__main__":
    # load data
    # simulation_data_location = "processed_data\\test\Belt_1_d.npz"
    simulation_data_location = "processed_data\\val\Belt_3.npz"
    simulation_data = np.load(simulation_data_location)
    sim_dataset = SimulationDataset(simulation_data)
    sim_data_loader = DataLoader(dataset=sim_dataset, num_workers=0, batch_size=1, shuffle=False)

    model = TCN(1 , 1, [32]*5, 5, 0).double()
    model.cuda()
    model.load_state_dict(torch.load("breath_Aligned_Array.pt"))
    eval_loss, model_prediction_list, ground_truth_list = run(sim_data_loader, name='run')
    draw_compare_result(model_prediction_list,ground_truth_list,zoom=[800,850])


