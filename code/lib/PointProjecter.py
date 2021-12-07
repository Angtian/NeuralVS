import torch
import torch.nn.functional as F
import math


all_parameters = ['azimuth', 'elevation', 'distance', 'focal', 'theta', 'principal', 'viewport']


class PointProjectPytorch(torch.nn.Module):
    def __init__(self, init_val, training_parameters, do_registration=True):
        super(PointProjectPytorch, self).__init__()
        for k in all_parameters:
            if k not in init_val.keys():
                raise Exception('Expected keys of initialization: %s but got: %s' % (str(all_parameters), str(init_val.keys())))

        for k in init_val.keys():
            if k not in all_parameters:
                pass
            requires_grad_ = bool(k in training_parameters)
            if k == 'principal':
                para_ = torch.nn.Parameter(torch.Tensor(list(init_val[k])).type(torch.float32), requires_grad=requires_grad_)
            else:
                para_ = torch.nn.Parameter(torch.Tensor([init_val[k]]).type(torch.float32), requires_grad=requires_grad_)

            if do_registration:
                self.register_parameter(k, para_)
            else:
                self.__setattr__(k, para_)

        self.prepare_transformation_matrix()

    def prepare_transformation_matrix(self):
        azimuth = -1 * self.azimuth
        elevation = self.elevation - math.pi / 2

        # [cos(azimuth), -sin(azimuth), 0],
        # [sin(azimuth), cos(azimuth), 0],
        # [0, 0, 1]
        Rz = torch.cat([torch.cos(azimuth), -torch.sin(azimuth), torch.zeros_like(azimuth),
                        torch.sin(azimuth), torch.cos(azimuth), torch.zeros_like(azimuth),
                        torch.zeros_like(azimuth), torch.zeros_like(azimuth), torch.ones_like(azimuth)
                        ]).view(3, 3)

        # [1, 0, 0],
        # [0, cos(elevation), -sin(elevation)],
        # [0, sin(elevation), cos(elevation)]
        Rx = torch.cat([torch.ones_like(elevation), torch.zeros_like(elevation), torch.zeros_like(elevation),
                        torch.zeros_like(elevation), torch.cos(elevation), -torch.sin(elevation),
                        torch.zeros_like(elevation), torch.sin(elevation), torch.cos(elevation),
                        ]).view(3, 3)

        R_rot = torch.mm(Rx, Rz)

        # [M * focal, 0, 0],
        # [0, M * focal, 0],
        # [0, 0, -1] M -> viewport
        Proj = torch.cat([self.viewport * self.focal, torch.zeros_like(self.focal), torch.zeros_like(self.focal),
                          torch.zeros_like(self.focal), self.viewport * self.focal, torch.zeros_like(self.focal),
                          torch.zeros_like(self.focal), torch.zeros_like(self.focal), -torch.ones_like(self.focal),
                          ]).view(3, 3)

        # R -> R.T
        self.proj = torch.t(torch.mm(Proj, R_rot))

        R2d = torch.cat([torch.cos(self.theta), -torch.sin(self.theta),
                        torch.sin(self.theta), torch.cos(self.theta),
                        ]).view(2, 2)
        trans = torch.Tensor([[1, 0], [0, -1]]).type(R2d.dtype).to(R2d.device)

        self.R2d = torch.mm(torch.t(R2d), trans)

        # C[0:3] = (distance * cos(elevation) * sin(azimuth), -distance * cos(elevation) * cos(azimuth)), distance * sin(elevation)
        # shape: (3, )
        self.bias2d = torch.cat([self.distance * torch.cos(elevation) * torch.sin(azimuth), -self.distance * torch.cos(elevation) * torch.cos(azimuth), self.distance * torch.sin(elevation)])

    def forward(self, x3d):
        # x3d @ R -> x2d , x2d[2]
        x2d = torch.mm(x3d, self.proj)

        # x2d[:, 0] = sum(x3d @ R[0, :].T + distance * cos(elevation) * sin(azimuth)) / sum(x3d @ R[2, :].T + distance * sin(elevation))
        # x2d[:, 1] = sum(x3d @ R[1, :].T - distance * cos(elevation) * cos(azimuth)) / sum(x3d @ R[2, :].T + distance * sin(elevation)
        x2d = (x2d + self.bias2d.view(1, 3))
        x2d = x2d[:, 0:2] / x2d[:, 2:]

        # x2d = x2d @ R2d.T @ [[1, 0], [0, -1]] (x2d[:, 1] *= -1)
        x2d = torch.mm(x2d, self.R2d)

        x2d += self.principal.view(1, 2)

        return x2d




