import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from model import VLSTM
from social_lstm_utils import *

a = torch.tensor([[[-0.0904,  0.0161,  0.0138,  0.0087, -0.0659,  0.1376, -0.1159,
          -0.0784, -0.0223, -0.0868,  0.0626, -0.0923,  0.1129, -0.0865,
          -0.0418, -0.0358,  0.1824, -0.1749, -0.0675, -0.1458,  0.0893,
           0.0436, -0.1239,  0.0597,  0.0947,  0.0970,  0.0824, -0.0509,
           0.0634,  0.0485,  0.1078, -0.0937,  0.0463,  0.1328, -0.1086,
          -0.2340,  0.0727, -0.0016, -0.1446, -0.0356, -0.0098, -0.0758,
          -0.0458, -0.1701,  0.1053, -0.0126,  0.1684, -0.0844,  0.0751,
           0.0306,  0.0843,  0.0505,  0.0027, -0.0864, -0.0467,  0.1991,
          -0.0424, -0.1183, -0.0684,  0.0107, -0.0568,  0.0686, -0.0721,
          -0.0053, -0.0414, -0.1361,  0.1364,  0.1879, -0.0378,  0.1109,
           0.0176, -0.0542, -0.0275,  0.0913, -0.1595,  0.1968,  0.0573,
           0.0741,  0.0044, -0.0572, -0.0835,  0.1486, -0.0987,  0.0068,
          -0.1532,  0.0290,  0.0496,  0.1125,  0.0427,  0.0992,  0.0978,
          -0.0978, -0.0143, -0.1220, -0.0675, -0.0646,  0.0914, -0.0221,
           0.0311,  0.0036,  0.0220,  0.0306,  0.1591, -0.1608, -0.0302,
          -0.0459, -0.2098, -0.0557,  0.0387,  0.1376, -0.0248,  0.0779,
           0.0917,  0.1828, -0.1363, -0.1145, -0.0333,  0.0495, -0.0875,
          -0.1719,  0.0373,  0.1376, -0.0170, -0.0062, -0.0712,  0.0229,
           0.0323,  0.0272]]], device='cuda:0', )   
b = torch.tensor([[[-0.2059,  0.0305,  0.0249,  0.0167, -0.1198,  0.2403, -0.2672,
          -0.1810, -0.0517, -0.1274,  0.1482, -0.2252,  0.2236, -0.2011,
          -0.0727, -0.0564,  0.3083, -0.3181, -0.1424, -0.2565,  0.1477,
           0.0668, -0.2752,  0.0976,  0.2268,  0.2549,  0.1425, -0.0858,
           0.1182,  0.1347,  0.1851, -0.2096,  0.1282,  0.2862, -0.1733,
          -0.3508,  0.1643, -0.0027, -0.2565, -0.0907, -0.0269, -0.1479,
          -0.0853, -0.3572,  0.1746, -0.0301,  0.3029, -0.1298,  0.1051,
           0.0609,  0.1410,  0.0986,  0.0053, -0.2036, -0.0946,  0.3282,
          -0.1513, -0.2668, -0.1328,  0.0202, -0.1072,  0.1523, -0.1524,
          -0.0152, -0.1056, -0.2746,  0.2827,  0.5016, -0.0646,  0.2115,
           0.0345, -0.1779, -0.0629,  0.2192, -0.2889,  0.3749,  0.1388,
           0.1732,  0.0089, -0.1458, -0.1613,  0.2204, -0.2898,  0.0135,
          -0.2823,  0.0628,  0.1287,  0.2292,  0.0887,  0.2158,  0.2379,
          -0.1863, -0.0245, -0.2212, -0.1186, -0.1275,  0.1954, -0.0441,
           0.0600,  0.0103,  0.0526,  0.0728,  0.4209, -0.3020, -0.0586,
          -0.0874, -0.3949, -0.1566,  0.0779,  0.2097, -0.0417,  0.1312,
           0.1544,  0.3854, -0.2551, -0.2123, -0.0762,  0.0837, -0.1500,
          -0.2759,  0.0697,  0.2496, -0.0257, -0.0184, -0.1629,  0.0458,
           0.0721,  0.0647]]], device='cuda:0')

class SLSTM(nn.Module):
    '''
    Class representing the Social LSTM model
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args: Training arguments
        '''
        super(SLSTM, self).__init__()
        self.embedded_input = args['embedded_input']
        self.embedding_occupancy_map = args['embedding_occupancy_map']
        self.use_speeds = args['use_speeds']
        self.grid_size = args['grid_size']
        self.max_dist = args['max_dist']
        self.hidden_size = args['hidden_size']
        model_checkpoint = args['trained_model']
        self.trained_model = VLSTM(args)
        if model_checkpoint is not None:
            if torch.cuda.is_available():
                load_params = torch.load(model_checkpoint)
            else:
                load_params = torch.load(
                    model_checkpoint, map_location=lambda storage, loc: storage)
            self.trained_model.load_state_dict(load_params['state_dict'])

            # Freeze model
            for param in self.trained_model.parameters():
                param.requires_grad = False

        self.embedding_spatial = nn.Linear(2, self.embedded_input)
        self.embedding_o_map = nn.Linear(
            (args['grid_size']**2) * self.hidden_size,
            self.embedding_occupancy_map)
        self.lstm = nn.LSTM(self.embedded_input +
                            self.embedding_occupancy_map, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 5)

    def forward(self, input_data, grids, neighbors, first_positions, num=12):
        selector = en_cuda(torch.LongTensor([2, 3]))
        #grids = Variable(grids)
        obs = 8
        # Embedd input and occupancy maps
        # Iterate over frames
        hiddens_neighb = [(autograd.Variable(en_cuda(torch.zeros(1, 1, self.hidden_size))), autograd.Variable(
            en_cuda(torch.zeros((1, 1, self.hidden_size))))) for i in range(len(neighbors[0]) + 1)]
        inputs = None
        temp = None
        for i in range(input_data.size()[0]):
            temp = input_data[i, :].unsqueeze(0)
            embedded_input = F.relu(self.embedding_spatial(
                temp[:, selector])).unsqueeze(0)
            # Iterate over peds if there is neighbors
            social_tensor = None

            # Check if the pedestrians has neighbors
            if len(neighbors[0]):
                all_frame = torch.cat(
                    [input_data[i, :].data.unsqueeze(0), neighbors[i]], 0)
                valid_indexes_center = grids[i][1]
                # Iterate over valid neighbors if exists

                if valid_indexes_center is not None:
                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes_center:
                        # Get hidden states from VLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        # print(hiddens_neighb[k+1][0],"  ",hiddens_neighb[k+1][1])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    if i>0:
                        last_obs = neighbors[i - 1][valid_indexes_center, :]
                        vlstm_in = neighbors[i][valid_indexes_center, :][:,[
                                    2, 3]] - last_obs[:,[2, 3]]
                        select = (last_obs[:,1] == -1).nonzero()
                        if len(select.size()):
                            vlstm_in[select.squeeze(1)] = 0

                    else:
                        vlstm_in = en_cuda(torch.zeros(len(valid_indexes_center),2))


                    # print(vlstm_in)
                    # print(Variable(vlstm_in))
                    # print(np.array(buff_hidden_c).shape)
                    if(np.array(buff_hidden_c).shape == (0,)):
                      buff_hidden_c.append(a)
                    buff_hidden_c = torch.cat(buff_hidden_c,1).detach().cpu().numpy()
                    # print(buff_hidden_c.shape)
                    
                    # print(np.array(buff_hidden_h).shape)
                    if(np.array(buff_hidden_h).shape == (0,)):
                      buff_hidden_h.append(b)
                    buff_hidden_h = torch.cat(buff_hidden_h,1).detach().cpu().numpy()
                    # print(buff_hidden_h.shape)
                    
                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.tensor(buff_hidden_c , device = 'cuda:0'),torch.tensor(buff_hidden_h, device = 'cuda:0')))

                    for idx,k in enumerate(valid_indexes_center):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes_center]

                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=grids[i][0], grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            embedded_o_map = F.relu(
                self.embedding_o_map(social_tensor)).unsqueeze(0)
            inputs = torch.cat([embedded_input, embedded_o_map], 2)
            if i == (obs - 1):
                break
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

        # Predict

        last = inputs

        results = []
        # Generate outputs
        points = []
        first_positions_c = first_positions.clone()
        for i in range(num):
            # get gaussian params for every point in batch
            hiddens_to_feed = hiddens_neighb[0]
            self.lstm.flatten_parameters()
            out, hidden = self.lstm(inputs, hiddens_to_feed)
            hiddens_neighb[0] = hidden

            linear_out = self.output(out.squeeze(0))

            linear_out = linear_out.split(1, 1)
            res_params = get_coef(*linear_out)
            results.append(torch.cat(res_params, 1).unsqueeze(0))
            res_params = torch.cat(res_params, 1)

            last_speeds = []
            last_grids = []
            temp_points = []
            mux = res_params.data[0, 0].detach().cpu().numpy()
            muy = res_params.data[0, 1].detach().cpu().numpy()
            sx = res_params.data[0, 2].detach().cpu().numpy()
            sy = res_params.data[0, 3].detach().cpu().numpy()
            rho = res_params.data[0, 4].detach().cpu().numpy()
            # Sample speeds
            speed = en_cuda(torch.Tensor(
                sample_gaussian_2d(mux, muy, sx, sy, rho)))
            pts = torch.add(speed, first_positions_c[0, :])
            first_positions_c[0, :] = pts

            # Compute embeddings  and social grid

            last_speeds = speed.unsqueeze(0)
            last_speeds = Variable(last_speeds)
            pts_frame = pts.unsqueeze(0)

            # SOCIAL TENSOR
            # Check if neighbors exists
            if(len(neighbors[i + obs])):
                pts_w_metadata = en_cuda(torch.Tensor(
                    [[neighbors[i + obs][0, 0], input_data[0, 1].data, pts[0], pts[1]]]))
                frame_all = torch.cat(
                    [pts_w_metadata, neighbors[i + obs]], 0)
                # Get positions in social_grid
                (indexes_in_grid, valid_indexes) = get_grid_positions(neighbors[i + obs], None, ped_data=pts_frame.squeeze(0),
                                                                      grid_size=self.grid_size, max_dist=self.max_dist)
                if(valid_indexes is not None):

                    buff_hidden_c,buff_hidden_h = [],[]
                    for k in valid_indexes:
                        # Get hidden states from OLSTM
                        buff_hidden_c.append(hiddens_neighb[k+1][0])
                        buff_hidden_h.append(hiddens_neighb[k+1][1])

                    # Compute neighbor speed
                    last_obs = neighbors[i + obs - 1][valid_indexes, :]
                    vlstm_in = neighbors[i+ obs][valid_indexes, :][:,[
                                2, 3]] - last_obs[:,[2, 3]]
                    select = (last_obs[:,1] == -1).nonzero()
                    if len(select.size()):
                        vlstm_in[select.squeeze(1)] = 0


                    hiddens_tmp = self.trained_model.get_hidden_states(
                        Variable(vlstm_in), (torch.cat(buff_hidden_c,1),torch.cat(buff_hidden_h,1)))

                    for idx,k in enumerate(valid_indexes):
                        hiddens_neighb[k+1] = (hiddens_tmp[0][:,idx,:].unsqueeze(0),hiddens_tmp[1][:,idx,:].unsqueeze(0))

                    # Compute social tensor
                    hidden_relu = [F.relu(hiddens_neighb[e + 1][0])
                                   for e in valid_indexes]
                    social_tensor = Variable(get_social_tensor(
                        hidden_relu, positions=indexes_in_grid, grid_size=self.grid_size))
                else:
                    social_tensor = Variable(en_cuda(torch.zeros(
                        1, (self.grid_size**2) * self.hidden_size)))
            else:
                social_tensor = Variable(en_cuda(torch.zeros(
                    1, (self.grid_size**2) * self.hidden_size)))

            last_speeds = F.relu(
                self.embedding_spatial(last_speeds)).unsqueeze(0)
            last_grids = F.relu(self.embedding_o_map(
                social_tensor)).unsqueeze(0)
            last = torch.cat([last_speeds, last_grids], 2)
            points.append(pts_frame.unsqueeze(0))

        results = torch.cat(results, 0)
        points = torch.cat(points, 0)
        return results, points
