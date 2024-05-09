from .tensorBase import *


class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        raise NotImplementedError


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        raise NotImplementedError


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        if self.block is None:
            self.density_plane, self.density_line = self.init_one_svd(
                self.density_n_comp, self.gridSize, 0.1, device, 1.0
            )
            self.app_plane, self.app_line = self.init_one_svd(
                self.app_n_comp, self.gridSize, 0.1, device, 0.0
            )
        else:
            N_batch = self.block.reindex_batch_1d_global[-1]
            N_voxel = self.block.N_voxels_in_batch
            n_scale = N_batch
            if self.opt is not None and self.opt.demo_mode_render:
                # N_voxel = self.opt.voxel_num_final[0]
                # fixme: hardcode here
                N_voxel = 128
            self.density_plane, self.density_line = self.init_kd_svd(
                N_batch, N_voxel, self.density_n_comp, 0.1, device, 0.5
            )
            self.app_plane, self.app_line = self.init_kd_svd(
                N_batch, N_voxel, self.app_n_comp, 2.0, device, 0.0
            )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)

    def init_kd_svd(self, n_batch, n_voxel, n_component, scale, device, bias=0.5):
        plane_coef, line_coef = [], []
        for i in range(3):
            B = int(n_batch)
            V = int(n_voxel)
            C = int(n_component[i])
            # line_c = torch.arange(1, C * B * V + 1, device=device, dtype=torch.float) #/ (B * V)
            # plane_c = torch.arange(1, C * B * V * V + 1, device=device, dtype=torch.float) #/ (B * V * V)
            # line_c = line_c.reshape(1, C, B, V)
            # plane_c = plane_c.reshape(1, C, B, V, V)
            plane_c = torch.randn([1, C, B, V, V], device=device)
            line_c = torch.randn([1, C, B, V], device=device)

            plane_c = (plane_c - plane_c.min()) / (plane_c.max() - plane_c.min())
            line_c = (line_c - line_c.min()) / (line_c.max() - line_c.min())
            plane_c = scale * plane_c + bias
            line_c = scale * line_c + bias
            # todo todo debug
            # plane_c = torch.ones([1, C, B, V, V], device=device)
            # line_c = torch.ones([1, C, B, V], device=device)
            # line_c = torch.arange(1, C * B * V + 1, device=device, dtype=torch.float) / (B * V * C)
            # plane_c = torch.arange(1, C * B * V * V + 1, device=device, dtype=torch.float) / (B * V * V * C)
            # line_c = line_c.reshape(1, C, B, V)
            # plane_c = plane_c.reshape(1, C, B, V, V)
            # line_c = torch.arange(1, B+1, device=device, dtype=torch.float) / B
            # plane_c = torch.arange(1, B+1, device=device, dtype=torch.float) / B
            # line_c = line_c.view(1,1,B,1).expand(1, C, B, V).clone()
            # plane_c = plane_c.view(1,1,B,1,1).expand(1, C, B, V, V).clone()
            plane_coef.append(torch.nn.Parameter(plane_c))
            line_coef.append(torch.nn.Parameter(line_c))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device, bias=1.0):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                )
            )
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
                )
            )

        return (
            torch.nn.ParameterList(plane_coef).to(device) + bias,
            torch.nn.ParameterList(line_coef).to(device) + bias,
        )

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars = [
            {"params": self.density_line, "lr": lr_init_spatialxyz},
            {"params": self.density_plane, "lr": lr_init_spatialxyz},
            {"params": self.app_line, "lr": lr_init_spatialxyz},
            {"params": self.app_plane, "lr": lr_init_spatialxyz},
            {"params": self.basis_mat.parameters(), "lr": lr_init_network},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [
                {"params": self.renderModule.parameters(), "lr": lr_init_network}
            ]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0

        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]

            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total
                + torch.mean(torch.abs(self.density_plane[idx]))
                + torch.mean(torch.abs(self.density_line[idx]))
            )  # + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = (
                total + reg(self.density_plane[idx]) * 1e-2
            )  # + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = (
                total + reg(self.app_plane[idx]) * 1e-2
            )  # + reg(self.app_line[idx]) * 1e-3
        return total

    def _compute_interpolated_feature_direct(
        self, xyz_sampled, feature_plane, feature_line, detach
    ):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]],
                xyz_sampled[..., self.matMode[1]],
                xyz_sampled[..., self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        if detach:
            coordinate_plane = coordinate_plane.detach()
            coordinate_line = coordinate_line.detach()

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(feature_plane)):
            plane_coef_point.append(
                F.grid_sample(
                    feature_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            line_coef_point.append(
                F.grid_sample(
                    feature_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True,
                ).view(-1, *xyz_sampled.shape[:1])
            )
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )
        return plane_coef_point, line_coef_point

    def _compute_interpolated_feature_kd(
        self, xyz_sampled, feature_plane, feature_line, detach
    ):
        invaabbSize = 2.0 / (self.aabb[1] - self.aabb[0])
        xyz_sampled = (xyz_sampled + 1) / invaabbSize + self.aabb[0]
        plane_coef_point, line_coef_point = self.block.query(
            xyz_sampled, feature_plane, feature_line, self.matMode, self.vecMode, detach
        )
        return plane_coef_point, line_coef_point

    def compute_interpolated_feature(
        self, xyz_sampled, feature_plane, feature_line, detach
    ):
        if self.block is None:
            func = self._compute_interpolated_feature_direct
        else:
            func = self._compute_interpolated_feature_kd
        return func(xyz_sampled, feature_plane, feature_line, detach)

    def compute_densityfeature(self, xyz_sampled):
        n_layer, n_point = len(self.density_plane), xyz_sampled.shape[0]
        plane_coef_point, line_coef_point = self.compute_interpolated_feature(
            xyz_sampled, self.density_plane, self.density_line, detach=False
        )

        if self.opt.density_mode == 'abs':
            plane_coef_point = torch.abs(plane_coef_point)
            line_coef_point = torch.abs(line_coef_point)
        elif self.opt.density_mode == 'square':
            plane_coef_point = plane_coef_point ** 2
            line_coef_point = line_coef_point ** 2
        elif self.opt.density_mode == 'none':
            pass
        else:
            raise NotImplementedError

        plane_coef_point = plane_coef_point.reshape(n_layer, -1, n_point)
        line_coef_point = line_coef_point.reshape(n_layer, -1, n_point)
        sigma_feature = torch.zeros((n_point,), device=xyz_sampled.device)
        for idx_plane in range(n_layer):
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point[idx_plane] * line_coef_point[idx_plane], dim=0
            )
        # sigma_feature = sigma_feature * 100.
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        plane_coef_point, line_coef_point = self.compute_interpolated_feature(
            xyz_sampled, self.app_plane, self.app_line, detach=True
        )
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            if self.block is None:
                plane_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[i].data,
                        size=(res_target[mat_id_1], res_target[mat_id_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                line_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[i].data,
                        size=(res_target[vec_id], 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            else:
                B = plane_coef[i].shape[2]
                V = res_target
                plane = plane_coef[i].data
                # plane[0, :3].view(3, 56*32, 32).permute(1, 2, 0).detach().cpu().numpy()
                plane = F.interpolate(
                    plane,
                    size=(B, V, V),
                    # mode='nearest',
                    mode="trilinear",
                    align_corners=True,
                )
                # plane = plane_coef[i].data.permute(0, 2, 1, 3, 4)[0]
                # plane = F.interpolate(
                #     plane,
                #     size=(V, V),
                #     # mode='nearest',
                #     mode="bilinear",
                #     align_corners=True,
                # )
                # plane = plane[None, ...].permute(0, 2, 1, 3, 4)
                plane_coef[i] = torch.nn.Parameter(plane)
                # plane[0, :3].permute(1, 2, 0).detach().cpu().numpy()
                line = line_coef[i].data
                line = F.interpolate(
                    line,
                    size=(B, V),
                    # mode='nearest',
                    mode="bilinear",
                    align_corners=True,
                )
                # line = line_coef[i].data.permute(0, 2, 1, 3)[0]
                # line = F.interpolate(
                #     line,
                #     size=V,
                #     # mode='nearest',
                #     mode="linear",
                #     align_corners=True,
                # )
                # line = line[None, ...].permute(0, 2, 1, 3)
                line_coef[i] = torch.nn.Parameter(line)

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # raise NotImplementedError
        self.block.upsample(res_target)
        self.app_plane, self.app_line = self.up_sampling_VM(
            self.app_plane, self.app_line, res_target
        )
        self.density_plane, self.density_line = self.up_sampling_VM(
            self.density_plane, self.density_line, res_target
        )

        res_target = [res_target, res_target, res_target]
        self.update_stepSize(res_target)
        print(f"upsamping to {res_target} =================== ")

    @torch.no_grad()
    def shrink(self, new_aabb):
        raise NotImplementedError
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
