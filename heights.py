from scipy import special
import random as rand
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import mpmath
import torch

import scipy
import hyper_geo
import util
import torch.nn as nn

def pytorch_average_plane_optimize_height(x, args, term = 10, height_lr = 1e-8, window_size = 3):
    
    au, hu, av, hv, t_1, t_2 = args

    h_params = []
    for h_init in x:
        h_params.append({'params': torch.nn.Parameter(torch.tensor([h_init]).double()), 'lr': height_lr})

    #print(h_params)
    optimizer = torch.optim.SGD(
        h_params
    , weight_decay=0.0)

    error_array = []
    
    for i in range(term):

        c1_array = []
        c2_array = []
        c3_array = []
        c4_array = []
        c5_array = []

        p_2d = []
        point_3d = []

        h_array = []

        head_2d_array = []

        #print(i, " HELLOOOOOOOOOOOOOOO")
        f_array = []
        for j in range(len(h_params) - window_size + 1):

            ankle_2d_w1 = torch.tensor([au[j:j + window_size][0], av[j:j + window_size][0], 1.0]).double()
            ankle_2d_w2 = torch.tensor([au[j:j + window_size][1], av[j:j + window_size][1], 1.0]).double()
            ankle_2d_w3 = torch.tensor([au[j:j + window_size][2], av[j:j + window_size][2], 1.0]).double()

            head_2d_w1 = torch.tensor([hu[j:j + window_size][0], hv[j:j + window_size][0], 1.0]).double()
            head_2d_w2 = torch.tensor([hu[j:j + window_size][1], hv[j:j + window_size][1], 1.0]).double()
            head_2d_w3 = torch.tensor([hu[j:j + window_size][2], hv[j:j + window_size][2], 1.0]).double()
            
            head_2d_array.append(head_2d_w1)
            head_2d_array.append(head_2d_w2)
            head_2d_array.append(head_2d_w3)

            h1 = h_params[j:j + window_size][0]['params'][0]
            h2 = h_params[j:j + window_size][1]['params'][0]
            h3 = h_params[j:j + window_size][2]['params'][0]

            h_array.append(h1)
            h_array.append(h2)
            h_array.append(h3)

            c1, c2, c3, c4, c5 = coef([au[j:j + window_size], hu[j:j + window_size]], [av[j:j + window_size], hv[j:j + window_size]], t_1, t_2, [h1, h2, h3]) # compute camera parameters

            f_squared = ((-c1*(c4*(au[j:j + window_size][0] - t_1) - c5*(au[j:j + window_size][1] - t_1)) - c2*(c4*(av[j:j + window_size][0] - t_2) - c5*(av[j:j + window_size][1] - t_2)))/(c3*(c4 - c5)))
            f = torch.sqrt(torch.abs(f_squared))

            n1 = c1
            n2 = c2
            n3 = f*c3

            n = torch.squeeze(torch.stack([n1, n2, n3]))

            lda = torch.norm(n)
            n = n/lda

            z1 = (f*c4/lda)
            z2 = (f*c5/lda)
            z3 = (-1*f/lda)
            #print(f, torch.zeros(1), torch.tensor([t_1]), " f asdadasdasd")
            torch.stack([f, torch.zeros(1).double(), torch.tensor([t_1]).double()])
            torch.stack([ torch.stack([f, torch.zeros(1).double(), torch.tensor([t_1]).double()]),  torch.stack([torch.zeros(1).double(), f, torch.tensor([t_2]).double()]),  torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])])

            #cam_matrix = torch.squeeze(torch.stack([ torch.stack([f, torch.zeros(1), torch.tensor([t_1])]),  torch.stack([torch.zeros(1), f, torch.tensor([t_2])]),  torch.stack([torch.zeros(1), torch.zeros(1), torch.ones(1)])])).double()
            cam_matrix = torch.squeeze(torch.stack([ torch.stack([f, torch.zeros(1).double(), torch.tensor([t_1]).double()]),  torch.stack([torch.zeros(1).double(), f, torch.tensor([t_2]).double()]),  torch.stack([torch.zeros(1).double(), torch.zeros(1).double(), torch.ones(1).double()])]))
            
            #print(cam_matrix, " cam matrxxxx")
            cam_inv = torch.inverse(cam_matrix) 
            
            ankle_3d1 = (cam_inv @ ankle_2d_w1)*torch.abs(z1)
            ankle_3d2 = (cam_inv @ ankle_2d_w2)*torch.abs(z2)
            ankle_3d3 = (cam_inv @ ankle_2d_w3)*torch.abs(z3)

            point_3d.append(torch.unsqueeze(ankle_3d1, dim = 0))
            point_3d.append(torch.unsqueeze(ankle_3d2, dim = 0))
            point_3d.append(torch.unsqueeze(ankle_3d3, dim = 0))

            f_array.append(f)
            p_2d.append(ankle_2d_w1)
            p_2d.append(ankle_2d_w2)
            p_2d.append(ankle_2d_w3)
        ###########
        '''
        points = torch.transpose(torch.cat(point_3d), 0, 1).detach().numpy()

        print(points, " points")
        print(points.shape, " POINTSSSS SHAPE")
        svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

        print(svd, " the svd")
        # Extract the left singular vectors
        left = svd[0]

        normal = left[:, -1]

        normal = normal/np.linalg.norm(normal)
        print(normal)
        '''
        ###########

        # subtract out the centroid and take the SVD

        points = torch.transpose(torch.cat(point_3d), 0, 1)

        plane_center = torch.mean(points, dim=1)

        svd = torch.svd(points - torch.unsqueeze(plane_center, dim = 1).repeat(1, points.shape[1]))
        
        # Extract the left singular vectors
        left = svd[0]

        # the corresponding left singular vector is the normal vector of the best-fitting plane

        normal = left[:, -1]

        normal = normal/torch.norm(normal)

        p_3d = util.plane_proj(normal, points, plane_center)

        fx, fy = util.ankle_calibration(p_2d, p_3d, t_1, t_2)

        #print(fx, torch.zeros(1), torch.tensor(t_1), " hiiiasdasdasdads")
        #print(torch.unsqueeze(fx, dim = 0), " fx")
        #torch.stack([fx, torch.zeros(1), torch.tensor(t_1)])
        #torch.stack([torch.zeros(1), fy, torch.tensor(t_2)])
        #torch.stack([torch.zeros(1), torch.zeros(1), torch.ones(1)])
        ######
        #print(fx, fy, " hellooo f")
        cam_matrix = torch.squeeze(torch.stack([ torch.stack([torch.unsqueeze(fx, dim = 0), torch.zeros(1), torch.tensor([t_1])]),  torch.stack([torch.zeros(1), torch.unsqueeze(fy, dim = 0), torch.tensor([t_2])]),  torch.stack([torch.zeros(1), torch.zeros(1), torch.ones(1)])])).double()
        cam_inv = torch.inverse(cam_matrix) 
        ######
        #print(p_3d.shape, torch.stack(h_array).shape, torch.unsqueeze(normal, dim = 0).repeat(len(h_array), 1).shape, " shapes")
        heads_3d = torch.squeeze(torch.transpose(p_3d, 0, 1) + torch.mul(torch.unsqueeze(torch.stack(h_array), dim = 0), torch.unsqueeze(normal, dim = 0).repeat(len(h_array), 1)))
        
        head_2d = (cam_matrix @ torch.transpose(heads_3d, 0, 1))
        
        head_2d = torch.div(head_2d[:2, :], head_2d[2, :]) 

        #print(torch.stack(head_2d_array).shape, " STACK H ARRAY")
        error_head = torch.transpose(torch.stack(head_2d_array), 0, 1)[:2, :] - head_2d
        '''
        print("********************")
        print(head_2d)
        print(torch.transpose(torch.stack(head_2d_array), 0, 1)[:2, :])
        print("********************")
        print(head_2d.shape, " SHAPEEEEEEE")
        print(heads_3d)
        print(heads_3d.shape, " heads_3d.shapeee")
        print(error_head, " errpor head")
        '''
        #COMPUTE FOCAL LENGTH HERE FROM THE PLANE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # PROBABLY HAVE TO SOLVE FOR THE FOCAL LENGTH GIVEN PLANE


        #print("before error comp")
        
        loss = nn.MSELoss()
        
        error = loss(head_2d, torch.transpose(torch.stack(head_2d_array), 0, 1)[:2, :])

        error_array.append(error)
        #error = torch.mean(torch.stack(error_head), dim=0)
        optimizer.zero_grad()
        error.backward(retain_graph=True)

        optimizer.step()

        with torch.no_grad():
            for p in range(len(h_params)):
                h_params[p]['params'][0].clamp_(0, 2.5)

        #print("**************************")
        #print(error, torch.mean(torch.stack(f_array), dim=0), " error")


        #print(h_params, " h params")
        #print(error_array[0].item(), "errror")
        #stop
    #print(f_array, " f _ arrat")
    
    h_return = []

    for p in range(len(h_params)):
        h_return.append(h_params[p]['params'][0].item())

    #print(h_return, " h return")
    return h_return, error_array[0].item()

def pytorch_average_optimize_height(x, args, term = 10, height_lr = 0.001, window_size = 3):
    
    au, hu, av, hv, t_1, t_2 = args

    h_params = []
    for h_init in x:
        h_params.append({'params': torch.nn.Parameter(torch.tensor([h_init]).double()), 'lr': height_lr})

    #print(h_params)
    optimizer = torch.optim.SGD(
        h_params
    , weight_decay=0.0)

    c1_array = []
    c2_array = []
    c3_array = []
    c4_array = []
    c5_array = []
    
    for i in range(term):
        #print(i, " HELLOOOOOOOOOOOOOOO")
        f_array = []
        for j in range(len(h_params) - window_size + 1):

            ankle_2d_w1 = torch.tensor([au[j:j + window_size][0], av[j:j + window_size][0], 1.0]).double()
            ankle_2d_w2 = torch.tensor([au[j:j + window_size][1], av[j:j + window_size][1], 1.0]).double()
            ankle_2d_w3 = torch.tensor([au[2], av[2], 1.0]).double()

            head_2d_w1 = torch.tensor([hu[j:j + window_size][0], hv[j:j + window_size][0], 1.0]).double()
            head_2d_w2 = torch.tensor([hu[j:j + window_size][1], hv[j:j + window_size][1], 1.0]).double()
            head_2d_w3 = torch.tensor([hu[j:j + window_size][2], hv[j:j + window_size][2], 1.0]).double()

            h1 = h_params[j:j + window_size][0]['params'][0]
            h2 = h_params[j:j + window_size][1]['params'][0]
            h3 = h_params[j:j + window_size][2]['params'][0]

            c1, c2, c3, c4, c5 = coef([au[j:j + window_size], hu[j:j + window_size]], [av[j:j + window_size], hv[j:j + window_size]], t_1, t_2, [h1, h2, h3]) # compute camera parameters

            c1_array.append(c1)
            c2_array.append(c2)
            c3_array.append(c3)
            c4_array.append(c4)
            c5_array.append(c5)

            f_squared = ((-c1*(c4*(au[j:j + window_size][0] - t_1) - c5*(au[j:j + window_size][1] - t_1)) - c2*(c4*(av[j:j + window_size][0] - t_2) - c5*(av[j:j + window_size][1] - t_2)))/(c3*(c4 - c5)))
            f = torch.sqrt(torch.absolute(f_squared))

            f_array.append(f)

    
        error_array = []
        for j in range(len(h_params) - window_size + 1):

            ankle_2d_w1 = torch.tensor([au[j:j + window_size][0], av[j:j + window_size][0], 1.0]).double()
            ankle_2d_w2 = torch.tensor([au[j:j + window_size][1], av[j:j + window_size][1], 1.0]).double()
            ankle_2d_w3 = torch.tensor([au[2], av[2], 1.0]).double()

            head_2d_w1 = torch.tensor([hu[j:j + window_size][0], hv[j:j + window_size][0], 1.0]).double()
            head_2d_w2 = torch.tensor([hu[j:j + window_size][1], hv[j:j + window_size][1], 1.0]).double()
            head_2d_w3 = torch.tensor([hu[j:j + window_size][2], hv[j:j + window_size][2], 1.0]).double()

            h1 = h_params[j:j + window_size][0]['params'][0]
            h2 = h_params[j:j + window_size][1]['params'][0]
            h3 = h_params[j:j + window_size][2]['params'][0]

            f = torch.mean(torch.stack(f_array), dim=0)
            
            #c1, c2, c3, c4, c5 = coef([au[j:j + window_size], hu[j:j + window_size]], [av[j:j + window_size], hv[j:j + window_size]], t_1, t_2, [h1, h2, h3])
            c1 = c1_array[j] 
            c2 = c2_array[j] 
            c3 = c3_array[j] 
            c4 = c4_array[j] 
            c5 = c5_array[j] 
            
            n1 = c1
            n2 = c2
            n3 = f*c3

            n = torch.squeeze(torch.stack([n1, n2, n3]))

            lda = torch.norm(n)
            n = n/lda

            z1 = (f*c4/lda)
            z2 = (f*c5/lda)
            z3 = (-1*f/lda)

            #####################
            cam_matrix = torch.squeeze(torch.stack([ torch.stack([f, torch.zeros(1), torch.tensor([t_1])]),  torch.stack([torch.zeros(1), f, torch.tensor([t_2])]),  torch.stack([torch.zeros(1), torch.zeros(1), torch.ones(1)])])).double()

            cam_inv = torch.inverse(cam_matrix) 
            
            ankle_3d1 = (cam_inv @ ankle_2d_w1)*torch.absolute(z1)
            ankle_3d2 = (cam_inv @ ankle_2d_w2)*torch.absolute(z2)
            ankle_3d3 = (cam_inv @ ankle_2d_w3)*torch.absolute(z3)

            head_3d1 = ankle_3d1 - n*h1
            head_3d2 = ankle_3d2 - n*h2
            head_3d3 = ankle_3d3 - n*h3

            head_pred1 = cam_matrix @ head_3d1
            head_pred2 = cam_matrix @ head_3d2
            head_pred3 = cam_matrix @ head_3d3 

            head_pred_2d1 = head_pred1[0:2]/head_pred1[2]
            head_pred_2d2 = head_pred2[0:2]/head_pred2[2]
            head_pred_2d3 = head_pred3[0:2]/head_pred3[2]

            error1 = head_pred_2d1 - head_2d_w1[0:2]
            error2 = head_pred_2d2 - head_2d_w2[0:2]
            error3 = head_pred_2d3 - head_2d_w3[0:2]

            #error_array.append((torch.norm(error1)**2 + torch.norm(error2)**2 + torch.norm(error3)**2))
            error_array.append(torch.norm(error1)**2)
            error_array.append(torch.norm(error2)**2)
            error_array.append(torch.norm(error3)**2)

        #print("before error comp")
        error = torch.mean(torch.stack(error_array), dim=0)
        optimizer.zero_grad()
        error.backward(retain_graph=True)

        optimizer.step()
        print(error, torch.mean(torch.stack(f_array), dim=0), " error")

    #print(f_array, " f _ arrat")
    return h_params

def pytorch_optimize_height(x, args, term = 10, height_lr = 0.1, window_size = 3):
    
    au, hu, av, hv, t_1, t_2 = args

    h_params = []
    for h_init in x:
        h_params.append({'params': torch.nn.Parameter(torch.tensor([h_init]).double()), 'lr': height_lr})

    #print(h_params)
    optimizer = torch.optim.SGD(
        h_params
    , weight_decay=0.0)
    
    for i in range(term):
        for j in range(len(h_params) - window_size + 1):

            ankle_2d_w1 = torch.tensor([au[j:j + window_size][0], av[j:j + window_size][0], 1.0]).double()
            ankle_2d_w2 = torch.tensor([au[j:j + window_size][1], av[j:j + window_size][1], 1.0]).double()
            ankle_2d_w3 = torch.tensor([au[j:j + window_size][2], av[j:j + window_size][2], 1.0]).double()

            head_2d_w1 = torch.tensor([hu[j:j + window_size][0], hv[j:j + window_size][0], 1.0]).double()
            head_2d_w2 = torch.tensor([hu[j:j + window_size][1], hv[j:j + window_size][1], 1.0]).double()
            head_2d_w3 = torch.tensor([hu[j:j + window_size][2], hv[j:j + window_size][2], 1.0]).double()

            h1 = h_params[j:j + window_size][0]['params'][0]
            h2 = h_params[j:j + window_size][1]['params'][0]
            h3 = h_params[j:j + window_size][2]['params'][0]

            c1, c2, c3, c4, c5 = coef([au[j:j + window_size], hu[j:j + window_size]], [av[j:j + window_size], hv[j:j + window_size]], t_1, t_2, [h1, h2, h3]) # compute camera parameters

            f_squared = ((-c1*(c4*(au[j:j + window_size][0] - t_1) - c5*(au[j:j + window_size][1] - t_1)) - c2*(c4*(av[j:j + window_size][0] - t_2) - c5*(av[j:j + window_size][1] - t_2)))/(c3*(c4 - c5)))
            f = torch.sqrt(torch.absolute(f_squared))

            n1 = c1
            n2 = c2
            n3 = f*c3

            n = torch.squeeze(torch.stack([n1, n2, n3]))
            lda = torch.norm(n)
            n = n/lda

            z1 = (f*c4/lda)
            z2 = (f*c5/lda)
            z3 = (-1*f/lda)

            #####################
            cam_matrix = torch.squeeze(torch.stack([ torch.stack([f, torch.zeros(1), torch.tensor([t_1])]),  torch.stack([torch.zeros(1), f, torch.tensor([t_2])]),  torch.stack([torch.zeros(1), torch.zeros(1), torch.ones(1)])])).double()

            #print(cam_matrix.shape, " cam matrix")
            cam_inv = torch.inverse(cam_matrix) 
            
            ankle_3d1 = (cam_inv @ ankle_2d_w1)*torch.absolute(z1)
            ankle_3d2 = (cam_inv @ ankle_2d_w2)*torch.absolute(z2)
            ankle_3d3 = (cam_inv @ ankle_2d_w3)*torch.absolute(z3)

            #print(ankle_3d1.shape, " ANKELASEAS")
            #print(n.shape, " noralaa")
            head_3d1 = ankle_3d1 - n*h1
            head_3d2 = ankle_3d2 - n*h2
            head_3d3 = ankle_3d3 - n*h3

            #print(head_3d1.shape, " head_3d1 ASDDASDAS")

            head_pred1 = cam_matrix @ head_3d1
            head_pred2 = cam_matrix @ head_3d2
            head_pred3 = cam_matrix @ head_3d3 

            head_pred_2d1 = head_pred1[0:2]/head_pred1[2]
            head_pred_2d2 = head_pred2[0:2]/head_pred2[2]
            head_pred_2d3 = head_pred3[0:2]/head_pred3[2]

            '''
            print(ankle_3d1, " ANKLES PRED")
            print(ankle_3d2, " ANKLES PRED")
            print(ankle_3d3, " ANKLES PRED")

            print(head_3d1, " Ahead PRED")
            print(head_3d2, " Ahead PRED")
            print(head_3d3, " Ahead PRED")
            '''

            #print(head_pred_2d1,  head_2d_w1[0:2])
            #print(head_pred_2d1.shape, head_2d_w1[0:2].shape)
            error1 = head_pred_2d1 - head_2d_w1[0:2]
            error2 = head_pred_2d2 - head_2d_w2[0:2]
            error3 = head_pred_2d3 - head_2d_w3[0:2]
            #print(head_pred_2d1, head_2d_w1[0:2], " head 1")
            #print(head_pred_2d2, head_2d_w2[0:2], " head 2")
            #print(head_pred_2d3, head_2d_w3[0:2], " head 3 12312123123")
            #print(torch.norm(error1), torch.norm(error2), torch.norm(error3), " errorsss")
            error = (torch.norm(error1) + torch.norm(error2) + torch.norm(error3))/3.0

            optimizer.zero_grad()
            error.backward(retain_graph=True)

            optimizer.step()
            print(error, f, " error")

    #print(f_array, " f _ arrat")
    return h_params

# This function implements method 1 in the report, the histogram method
def function_optimize_height(x, args):
    
    au, hu, av, hv, t_1, t_2 = args
    h1 = x[0]
    h2 = x[1]
    h3 = x[2]

    print(h1,h2,h3, " HEIGHTSSSS")
    
    ankle_2d_w1 = np.array([au[0], av[0], 1.0])
    ankle_2d_w2 = np.array([au[1], av[1], 1.0])
    ankle_2d_w3 = np.array([au[2], av[2], 1.0])

    head_2d_w1 = np.array([hu[0], hv[0], 1.0])
    head_2d_w2 = np.array([hu[1], hv[1], 1.0])
    head_2d_w3 = np.array([hu[2], hv[2], 1.0])

    
    print(ankle_2d_w1, " 2d ANKLES PRED")
    print(ankle_2d_w2, "2d  ANKLES PRED")
    print(ankle_2d_w3, " 2d ANKLES PRED")

    print(head_2d_w1, " 2d Ahead PRED")
    print(head_2d_w2, " 2d Ahead PRED")
    print(head_2d_w3, " 2d Ahead PRED")
    
    c1, c2, c3, c4, c5 = coef([au, hu], [av, hv], t_1, t_2, x) # compute camera parameters

    #c1 = c1/h3
    #c2 = c2/h3
    #c3 = c3/h3
    #c4 = c4*(h1/h3)
    #c5 = c5*(h2/h3)
    #f_squared = ((-c1*(c4*(au[0] - t_1) - c5*(au[1] - t_1))*h1 - c2*(c4*(av[0] - t_2) - c5*(av[1] - t_2))*h2)/(c3*(c4*h1 - c5*h2)))
    #f_squared = (((-c1*c4*(au[0] - t_1) - c2*c4*(av[1] - t_2))*h1 + (c1*c5*(hu[0] - t_1) + c2*c5*(hv[1] - t_2))*h2)/(c3*(c4*h1 - c5*h2)))
    #f_squared = ((-c1*(c4*(au[0] - t_1) - c5*(au[1] - t_1)) - c2*(c4*(av[0] - t_2) - c5*(av[1] - t_2)))/(c3*(c4 - c5)))
    f_squared = ((-c1*(c4*(au[0] - t_1) - c5*(au[1] - t_1)) - c2*(c4*(av[0] - t_2) - c5*(av[1] - t_2)))/(c3*(c4 - c5)))
    f = np.sqrt(np.absolute(f_squared))

    n1 = c1
    n2 = c2
    n3 = f*c3

    n = np.array([n1, n2, n3])
    lda = np.linalg.norm(n)
    n = n/lda

    #print(h1,h2,h3 , " HEIGHTSSSS")
    #print(n, " NORMALLLL")
    z1 = (f*c4/lda)
    z2 = (f*c5/lda)
    z3 = (-1*f/lda)

    #####################
    cam_matrix = np.array([[f, 0, t_1], [0, f, t_2], [0, 0, 1]])
    #print(z1,z2, z3, " depths")
    #print(f, " hiiii focal")
    cam_inv = np.linalg.inv(cam_matrix) 
    
    ankle_3d1 = (cam_inv @ ankle_2d_w1)*np.absolute(z1)
    ankle_3d2 = (cam_inv @ ankle_2d_w2)*np.absolute(z2)
    ankle_3d3 = (cam_inv @ ankle_2d_w3)*np.absolute(z3)

    head_3d1 = ankle_3d1 - n*h1
    head_3d2 = ankle_3d2 - n*h2
    head_3d3 = ankle_3d3 - n*h3

    head_pred1 = cam_matrix @ head_3d1
    head_pred2 = cam_matrix @ head_3d2
    head_pred3 = cam_matrix @ head_3d3 

    head_pred_2d1 = head_pred1[0:2]/head_pred1[2]
    head_pred_2d2 = head_pred2[0:2]/head_pred2[2]
    head_pred_2d3 = head_pred3[0:2]/head_pred3[2]

    
    print(ankle_3d1, " ANKLES PRED")
    print(ankle_3d2, " ANKLES PRED")
    print(ankle_3d3, " ANKLES PRED")

    print(head_3d1, " Ahead PRED")
    print(head_3d2, " Ahead PRED")
    print(head_3d3, " Ahead PRED")

    #print(head_pred_2d1,  head_2d_w1[0:2])
    error1 = head_pred_2d1 - head_2d_w1[0:2]
    error2 = head_pred_2d2 - head_2d_w2[0:2]
    error3 = head_pred_2d3 - head_2d_w3[0:2]

    error = (np.linalg.norm(error1) + np.linalg.norm(error2) + np.linalg.norm(error3))/3.0

    print(error, f, " error")
    stop
    #print(f_array, " f _ arrat")
    return error

def optimize_height(au, hu, av, hv, t_1, t_2, itr = 1):

    f_array = []
    n_array = []
    z1_array = []
    z2_array = []
    z3_array = []

    f_squared_array = []

    h1 = 1.6
    h2 = 1.6
    h3 = 1.6

    
    ankle_2d_w1 = np.array([au[0], av[0], 1.0])
    ankle_2d_w2 = np.array([au[1], av[1], 1.0])
    ankle_2d_w3 = np.array([au[2], av[2], 1.0])

    head_2d_w1 = np.array([hu[0], hv[0], 1.0])
    head_2d_w2 = np.array([hu[1], hv[1], 1.0])
    head_2d_w3 = np.array([hu[2], hv[2], 1.0])
    
    c1, c2, c3, c4, c5 = coef([au, hu], [av, hv], t_1, t_2) # compute camera parameters

    for i in range(itr):
        #c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3 = compute_focal([au, hu], [av, hv], t_1, t_2, [h1, h2, h3]) # compute camera parameters
        
        #####################
        f_squared = ((-c1*(c4*(au[0] - t_1) - c5*(au[1] - t_1)) - c2*(c4*(av[0] - t_2) - c5*(av[1] - t_2)))/(c3*(c4 - c5)))
        f = np.sqrt(np.absolute(f_squared))

        n1 = c1
        n2 = c2
        n3 = f*c3

        n = np.array([n1, n2, n3])
        lda = np.linalg.norm(n)
        n = n/lda

        z1 = f*c4/lda
        z2 = f*c5/lda
        z3 = -1*f/lda

        #####################
        cam_matrix = np.array([[f, 0, t_1], [0, f, t_2], [0, 0, 1]])

        cam_inv = np.linalg.inv(cam_matrix) 
        
        ankle_3d1 = (cam_inv @ ankle_2d_w1)*np.absolute(z1)
        ankle_3d2 = (cam_inv @ ankle_2d_w2)*np.absolute(z2)
        ankle_3d3 = (cam_inv @ ankle_2d_w3)*np.absolute(z3)

        head_3d1 = ankle_3d1 - n*h1
        head_3d2 = ankle_3d2 - n*h2
        head_3d3 = ankle_3d3 - n*h3

        error1 = cam_matrix @ head_3d1
        error2 = cam_matrix @ head_3d2
        error3 = cam_matrix @ head_3d3 

        error1 = error1[0:2]/error1[2] - head_2d_w1[0:2]
        error2 = error2[0:2]/error2[2] - head_2d_w2[0:2]
        error3 = error3[0:2]/error3[2] - head_2d_w3[0:2]

        error = (np.linalg.norm(error1) + np.linalg.norm(error2) + np.linalg.norm(error3))/3.0

        print(error, " error")

        f_squared_array.append(f_squared)
        f_array.append(f)
        n_array.append(n)
        z1_array.append(z1)
        z2_array.append(z2)
        z3_array.append(z3)

    #print(f_array, " f _ arrat")
    return f_array, f_squared_array, np.array(n_array), z1_array, z2_array, z3_array

#implements the random variable for focal length
def compute_focal(u, v, t_1, t_2, h):
    
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]
    #print(np.array(v).shape, " shape")
    v_11 = v[0][0]
    v_12 = v[0][1]
    v_13 = v[0][2]

    v_21 = v[1][0]
    v_22 = v[1][1] 
    v_23 = v[1][2]

    #########################

    u_11 = u[0][0] 
    u_12 = u[0][1]
    u_13 = u[0][2]

    u_21 = u[1][0]
    u_22 = u[1][1]
    u_23 = u[1][2]

    c1 = (-t_1*u_11*v_12*v_13 + t_1*u_11*v_12*v_23 + t_1*u_11*v_13*v_22 - t_1*u_11*v_22*v_23 + t_1*u_12*v_11*v_13 - t_1*u_12*v_11*v_23 - t_1*u_12*v_13*v_21 + t_1*u_12*v_21*v_23 + t_1*u_21*v_12*v_13 - t_1*u_21*v_12*v_23 - t_1*u_21*v_13*v_22 + t_1*u_21*v_22*v_23 - t_1*u_22*v_11*v_13 + t_1*u_22*v_11*v_23 + t_1*u_22*v_13*v_21 - t_1*u_22*v_21*v_23 + u_11*u_12*v_13*v_21 - u_11*u_12*v_13*v_22 - u_11*u_12*v_21*v_23 + u_11*u_12*v_22*v_23 + u_11*u_22*v_12*v_13 - u_11*u_22*v_12*v_23 - u_11*u_22*v_13*v_21 + u_11*u_22*v_21*v_23 - u_12*u_21*v_11*v_13 + u_12*u_21*v_11*v_23 + u_12*u_21*v_13*v_22 - u_12*u_21*v_22*v_23 + u_21*u_22*v_11*v_13 - u_21*u_22*v_11*v_23 - u_21*u_22*v_12*v_13 + u_21*u_22*v_12*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c2 = (-t_2*u_11*v_12*v_13 + t_2*u_11*v_12*v_23 + t_2*u_11*v_13*v_22 - t_2*u_11*v_22*v_23 + t_2*u_12*v_11*v_13 - t_2*u_12*v_11*v_23 - t_2*u_12*v_13*v_21 + t_2*u_12*v_21*v_23 + t_2*u_21*v_12*v_13 - t_2*u_21*v_12*v_23 - t_2*u_21*v_13*v_22 + t_2*u_21*v_22*v_23 - t_2*u_22*v_11*v_13 + t_2*u_22*v_11*v_23 + t_2*u_22*v_13*v_21 - t_2*u_22*v_21*v_23 + u_11*v_12*v_13*v_21 - u_11*v_12*v_21*v_23 - u_11*v_13*v_21*v_22 + u_11*v_21*v_22*v_23 - u_12*v_11*v_13*v_22 + u_12*v_11*v_22*v_23 + u_12*v_13*v_21*v_22 - u_12*v_21*v_22*v_23 - u_21*v_11*v_12*v_13 + u_21*v_11*v_12*v_23 + u_21*v_11*v_13*v_22 - u_21*v_11*v_22*v_23 + u_22*v_11*v_12*v_13 - u_22*v_11*v_12*v_23 - u_22*v_12*v_13*v_21 + u_22*v_12*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c3 = (u_11*v_12*v_13 - u_11*v_12*v_23 - u_11*v_13*v_22 + u_11*v_22*v_23 - u_12*v_11*v_13 + u_12*v_11*v_23 + u_12*v_13*v_21 - u_12*v_21*v_23 - u_21*v_12*v_13 + u_21*v_12*v_23 + u_21*v_13*v_22 - u_21*v_22*v_23 + u_22*v_11*v_13 - u_22*v_11*v_23 - u_22*v_13*v_21 + u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c4 = (-h1*u_12*v_13*v_21 + h1*u_12*v_13*v_22 + h1*u_12*v_21*v_23 - h1*u_12*v_22*v_23 + h1*u_21*v_12*v_13 - h1*u_21*v_12*v_23 - h1*u_21*v_13*v_22 + h1*u_21*v_22*v_23 - h1*u_22*v_12*v_13 + h1*u_22*v_12*v_23 + h1*u_22*v_13*v_21 - h1*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c5 = (-h2*u_11*v_13*v_21 + h2*u_11*v_13*v_22 + h2*u_11*v_21*v_23 - h2*u_11*v_22*v_23 + h2*u_21*v_11*v_13 - h2*u_21*v_11*v_23 - h2*u_21*v_13*v_22 + h2*u_21*v_22*v_23 - h2*u_22*v_11*v_13 + h2*u_22*v_11*v_23 + h2*u_22*v_13*v_21 - h2*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    
    f_squared = ((-c1*(c4*(u_11 - t_1) - c5*(u_12 - t_1)) - c2*(c4*(v_11 - t_2) - c5*(v_12 - t_2)))/(c3*(c4 - c5)))
    f = np.sqrt(np.absolute(f_squared))

    n1 = c1
    n2 = c2
    n3 = f*c3

    n = np.array([n1, n2, n3])
    lda = np.linalg.norm(n)
    n = n/lda

    z1 = f*c4/lda
    z2 = f*c5/lda
    z3 = -1*f/lda

    return c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3


#implements the random variable for focal length
def coef(u, v, t_1, t_2, h):
    
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]

    v_11 = v[0][0]
    v_12 = v[0][1]
    v_13 = v[0][2]

    v_21 = v[1][0]
    v_22 = v[1][1] 
    v_23 = v[1][2]

    #########################

    u_11 = u[0][0] 
    u_12 = u[0][1]
    u_13 = u[0][2]

    u_21 = u[1][0]
    u_22 = u[1][1]
    u_23 = u[1][2]

    c1 = (-t_1*u_11*v_12*v_13 + t_1*u_11*v_12*v_23 + t_1*u_11*v_13*v_22 - t_1*u_11*v_22*v_23 + t_1*u_12*v_11*v_13 - t_1*u_12*v_11*v_23 - t_1*u_12*v_13*v_21 + t_1*u_12*v_21*v_23 + t_1*u_21*v_12*v_13 - t_1*u_21*v_12*v_23 - t_1*u_21*v_13*v_22 + t_1*u_21*v_22*v_23 - t_1*u_22*v_11*v_13 + t_1*u_22*v_11*v_23 + t_1*u_22*v_13*v_21 - t_1*u_22*v_21*v_23 + u_11*u_12*v_13*v_21 - u_11*u_12*v_13*v_22 - u_11*u_12*v_21*v_23 + u_11*u_12*v_22*v_23 + u_11*u_22*v_12*v_13 - u_11*u_22*v_12*v_23 - u_11*u_22*v_13*v_21 + u_11*u_22*v_21*v_23 - u_12*u_21*v_11*v_13 + u_12*u_21*v_11*v_23 + u_12*u_21*v_13*v_22 - u_12*u_21*v_22*v_23 + u_21*u_22*v_11*v_13 - u_21*u_22*v_11*v_23 - u_21*u_22*v_12*v_13 + u_21*u_22*v_12*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c2 = (-t_2*u_11*v_12*v_13 + t_2*u_11*v_12*v_23 + t_2*u_11*v_13*v_22 - t_2*u_11*v_22*v_23 + t_2*u_12*v_11*v_13 - t_2*u_12*v_11*v_23 - t_2*u_12*v_13*v_21 + t_2*u_12*v_21*v_23 + t_2*u_21*v_12*v_13 - t_2*u_21*v_12*v_23 - t_2*u_21*v_13*v_22 + t_2*u_21*v_22*v_23 - t_2*u_22*v_11*v_13 + t_2*u_22*v_11*v_23 + t_2*u_22*v_13*v_21 - t_2*u_22*v_21*v_23 + u_11*v_12*v_13*v_21 - u_11*v_12*v_21*v_23 - u_11*v_13*v_21*v_22 + u_11*v_21*v_22*v_23 - u_12*v_11*v_13*v_22 + u_12*v_11*v_22*v_23 + u_12*v_13*v_21*v_22 - u_12*v_21*v_22*v_23 - u_21*v_11*v_12*v_13 + u_21*v_11*v_12*v_23 + u_21*v_11*v_13*v_22 - u_21*v_11*v_22*v_23 + u_22*v_11*v_12*v_13 - u_22*v_11*v_12*v_23 - u_22*v_12*v_13*v_21 + u_22*v_12*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c3 = (u_11*v_12*v_13 - u_11*v_12*v_23 - u_11*v_13*v_22 + u_11*v_22*v_23 - u_12*v_11*v_13 + u_12*v_11*v_23 + u_12*v_13*v_21 - u_12*v_21*v_23 - u_21*v_12*v_13 + u_21*v_12*v_23 + u_21*v_13*v_22 - u_21*v_22*v_23 + u_22*v_11*v_13 - u_22*v_11*v_23 - u_22*v_13*v_21 + u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c4 = (-h1*u_12*v_13*v_21 + h1*u_12*v_13*v_22 + h1*u_12*v_21*v_23 - h1*u_12*v_22*v_23 + h1*u_21*v_12*v_13 - h1*u_21*v_12*v_23 - h1*u_21*v_13*v_22 + h1*u_21*v_22*v_23 - h1*u_22*v_12*v_13 + h1*u_22*v_12*v_23 + h1*u_22*v_13*v_21 - h1*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c5 = (-h2*u_11*v_13*v_21 + h2*u_11*v_13*v_22 + h2*u_11*v_21*v_23 - h2*u_11*v_22*v_23 + h2*u_21*v_11*v_13 - h2*u_21*v_11*v_23 - h2*u_21*v_13*v_22 + h2*u_21*v_22*v_23 - h2*u_22*v_11*v_13 + h2*u_22*v_11*v_23 + h2*u_22*v_13*v_21 - h2*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)

    return c1, c2, c3, c4, c5
