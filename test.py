# test.py

import time
import torch
import torch.optim as optim
import torch.nn as nn
import evaluate
from torchvision.utils import save_image
import plugins
import numpy as np
from rendering import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import euler_angles_to_matrix
import utils
import morphology

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize_channelwise(x):
    x_min = x.min(dim=0, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min)

def denormalize(x):
    x_max = x.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return 2 * (x - x_max/2)

def threshold(x):
    x[x<0.5] = 0
    x[x>=0.5] = 1
    return x

def camera_matrix_to_rst(camera_mat):
    # Resolves Nx3x4 camera matrices into corresponding rotation, translation and scale
    translation = camera_mat[:,:2,3]
    scale = camera_mat[...,:3].norm(dim=2, keepdim=True)
    rotation = camera_mat[...,:3] / scale
    return rotation, scale, translation

class Tester:
    def __init__(self, args, model, modelO, modelR, criterion, renderer, tb_writer=None):
        self.args = args
        self.modelE, self.modelDS, self.modelDT = model[:3]
        self.use_conf = args.use_conf
        self.modelO = modelO
        self.modelR = modelR
        self.criterion = criterion

        self.save_results = args.save_results
        self.renderer = renderer

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.cuda = args.cuda
        self.ngpu = args.ngpu
        self.device = torch.device("cuda" if (self.cuda and torch.cuda.is_available()) else "cpu")
        self.devices = []
        for idx in range(self.ngpu):
            self.devices.append(torch.device("cuda:{}".format(idx)))

        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.texture_size = args.texture_size
        self.nchannels = args.nchannels
        self.vertex_num = args.vertex_num
        self.faces = self.renderer.tri[:,:-1].transpose(1,0).cpu().numpy().astype('int32')
        self.is_using_symmetry = args.is_using_symmetry
        self.dilation = morphology.Dilation2d(in_channels=1, out_channels=1, kernel_size=3, soft_max=False, device=self.device)

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            self.save_results
        )
        self.params_loss = ['ReconL2', 'SymLoss', 'ReconIL2', 'PSNR', 'SSIM', 'Loss']
        self.log_loss.register(self.params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'ReconL2': {'dtype': 'running_mean'},
            'SymLoss': {'dtype': 'running_mean'},
            'ReconIL2': {'dtype': 'running_mean'},
            'PSNR': {'dtype': 'running_mean'},
            'SSIM': {'dtype': 'running_mean'},
            'Loss': {'dtype': 'running_mean'}
        }
        self.monitor.register(self.params_monitor)

        # display testing progress
        if self.log_type == 'traditional':
            self.print_formatter = 'Test [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.3f "
        elif self.log_type == 'progressbar':
            self.print_formatter = '({}/{})'
            for item in self.params_loss:
                self.print_formatter += ' | ' + item + ' {:.2f}'

        self.tb_writer = tb_writer
        self.losses = {}
        self.setupParaStat()
        self.get_psnr = getattr(evaluate, 'PSNR')()
        self.get_ssim = getattr(evaluate, 'SSIM')()

    def setupParaStat(self):
        self.const_alb_mask = torch.tensor(load_const_alb_mask(), device=self.device).long()
        self.uv_tri, self.uv_mask = self.renderer.tri_2d, self.renderer.tri_2d_mask

        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')
        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=torch.float32).to(self.device)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), self.vertex_num), dtype=torch.float32).to(self.device)
        self.w_shape = torch.tensor(w_shape, dtype=torch.float32, device=self.device)
        self.w_exp = torch.tensor(w_exp, dtype=torch.float32, device=self.device)
        self.mean_m = torch.tensor(np.load('datasets/mean_m.npy'), dtype=torch.float32).to(self.device)
        self.std_m = torch.tensor(np.load('datasets/std_m.npy'), dtype=torch.float32).to(self.device)
        self.mean_tex = torch.tensor(np.load('datasets/mean_texture.npy'), dtype=torch.float32).to(self.device)
        self.shape_para_mean = torch.tensor(np.load('datasets/shape_para_mean.npy'), dtype=torch.float32).to(self.device)
        self.shape_para_std = torch.tensor(np.load('datasets/shape_para_std.npy'), dtype=torch.float32).to(self.device)

    def model_eval(self):
        self.modelE.eval()
        self.modelDS.eval()
        self.modelDT.eval()
        if self.modelO is not None:
            self.modelO.eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()

        # Switch 3DMM models to eval mode
        self.model_eval()
        self.modelR.eval()

        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Test'), max=len(dataloader))
        end = time.time()
        celeba_flist = open('celeba_test.flist', 'w')
        for i, (images, filenames) in enumerate(dataloader):            # keeps track of data loading time          # Faces
            data_time = time.time() - end
            batch_size = images.shape[0]
            images = images.to(self.device)

            with torch.no_grad():
                face_parsing_out = self.modelO(images).to(self.device)
                face_mask = threshold(face_parsing_out[:,0:1])
                occ_mask = threshold(face_parsing_out[:,1:2])
                background_mask = threshold(face_parsing_out[:,2:3])
                image_mask = 1 - background_mask        # face_mask for occlusion removal

            # create mask
            occl_gt = torch.ones_like(images)[:,0:1]
            row_o = torch.randint(images.shape[2]//4, images.shape[2]//2, (batch_size, 1)).squeeze()
            col_o = torch.randint(images.shape[3]//4, images.shape[3]//2, (batch_size, 1)).squeeze()
            height_o = torch.randint(images.shape[2]//3, images.shape[2]*3//4, (batch_size, 1)).squeeze()
            width_o = torch.randint(images.shape[3]//3, images.shape[3]*3//4, (batch_size, 1)).squeeze()
            for j in range(batch_size):
                occl_gt[j, :, row_o[j]:min(row_o[j]+height_o[j], images.shape[2]-1), col_o[j]:min(col_o[j]+width_o[j], images.shape[3]-1)] = 0
            occl_gt = (occl_gt.bool() + (1-image_mask).bool()).float()
            image_in = images * occl_gt

            ############################
            # Run the network
            ############################
            shape_feats, tex_feats, conf_feats, scale_out, pose_out, il_feats = self.modelE(image_in)
            shape_feats_unnorm = shape_feats * self.shape_para_std + self.shape_para_mean
            shape_out_full = torch.matmul(shape_feats_unnorm[:,:199], torch.transpose(self.w_shape, 1, 0)) + torch.matmul(shape_feats_unnorm[:,199:], torch.transpose(self.w_exp, 1, 0)) + self.mean_shape

            # Get full output pose
            rot_mat_out = euler_angles_to_matrix(pose_out[:,:3]*math.pi/2, convention="XYZ")
            translate_out = 0.5*(pose_out[:,3:]+1)*self.resolution[0]
            translate_out_ext = torch.cat((translate_out, torch.zeros(batch_size,1).to(self.device)), dim=1).unsqueeze(1)
            pose_mat_out = torch.cat((rot_mat_out.permute(0,2,1), translate_out_ext), dim=1).permute(0,2,1)
            pose_mat_out[:,:2,:3] *= scale_out.unsqueeze(-1).unsqueeze(-1)

            # Rendering
            vertex3d_out, vertex2d_out = self.renderer.get_projection(mshape=shape_out_full, m=pose_mat_out.permute(0,2,1))
            normal_out, normalf_out, m_normal_out, m_normalf_out = self.renderer.compute_normal(vertex3d_out, rot_mat=rot_mat_out)

            # Shade and Texture Output
            shade_uv = self.renderer.generate_shade(il_feats, m_normal_out)

            # Rendering parameters
            depth, verts_map, barycoords, trimap_mask = self.renderer.get_image_mapping_params_from_vt(vertex2d_out, m_normalf_out)
            depth = depth.unsqueeze(1)

            # Extract the texture and albedo of the input and original images and mask using the obtained shape and shade
            tex_in_vt, tex_in_vt_mask = self.renderer.get_vt_from_image(images, vertex2d_out, m_normalf_out, m_normal_out)
            occ_in_vt = self.renderer.get_vt_from_image(occl_gt, vertex2d_out, m_normalf_out, m_normal_out)[0]
            tex_mask_vt = self.renderer.get_vt_from_image(image_mask, vertex2d_out, m_normal=m_normal_out, m_normalf=m_normalf_out)[0]

            tex_gt_uv, rendering_mask = self.renderer.get_uv_from_vt(tex_in_vt, tex_in_vt_mask)
            occl_uv = threshold(self.renderer.get_uv_from_vt(occ_in_vt, tex_in_vt_mask)[0])
            tex_mask_uv = threshold(self.renderer.get_uv_from_vt(tex_mask_vt, tex_in_vt_mask)[0])
            occl_uv *= tex_mask_uv

            shade_uv[shade_uv < 0.5] = 0.5
            alb_gt_uv = normalize(tex_gt_uv) / (shade_uv + 1e-8)
            alb_gt_uv_max = alb_gt_uv.max()
            alb_gt_uv -= alb_gt_uv_max/2
            alb_in_uv = alb_gt_uv * occl_uv

            # Run the network
            alb_out_uv, (gate1, gate2, attn), conf_uv = self.modelR(alb_in_uv, mask=occl_uv)
            recon_loss = self.criterion.norm_loss(alb_out_uv, alb_gt_uv, mask=tex_mask_uv, loss_type='l1')*5

            # Symmetry loss
            if self.is_using_symmetry:
                alb_out_uv_flip = alb_out_uv.flip(dims=(3,)).detach()
                alb_diff_ignore = torch.abs(alb_out_uv - alb_out_uv_flip) <= 0.005
                alb_out_uv_flip[alb_diff_ignore] = alb_out_uv[alb_diff_ignore]
                symmetry_loss = self.criterion.norm_loss(alb_out_uv, alb_out_uv_flip, mask=(1-occl_uv)*self.uv_mask, loss_type='l1')*5

            # Reconstructed output
            tex_out_uv = (2*((alb_out_uv + alb_gt_uv_max/2) * shade_uv - 0.5)).clamp(-1, 1)
            tex_out_vt, tex_out_vt_mask = self.renderer.get_vt_from_uv(tex_out_uv)
            recon_output, recon_output_mask = self.renderer.get_image_from_mapping_params(tex_out_vt, tex_in_vt_mask, verts_map, barycoords, trimap_mask)

            conf_vt, _ = self.renderer.get_vt_from_uv(conf_uv)
            conf_img, _ = self.renderer.get_image_from_mapping_params(conf_vt, tex_in_vt_mask, verts_map, barycoords, trimap_mask)

            # Blended output
            blend_mask = image_mask * recon_output_mask.unsqueeze(1)
            occl_gt = self.dilation((occl_gt.bool() + (1 - blend_mask).bool()).float())
            complete_blended_output = images * (1 - blend_mask) + recon_output * blend_mask
            blended_output = images * occl_gt + recon_output * (1-occl_gt)

            # Image recon loss
            recon_loss_image = self.criterion.norm_loss(complete_blended_output, images, mask=image_mask, conf=conf_img, loss_type='l1')*5
            total_loss = recon_loss + symmetry_loss + recon_loss_image

            self.losses['ReconL2'] = recon_loss.item()
            self.losses['SymLoss'] = symmetry_loss.item()
            self.losses['ReconIL2'] = recon_loss_image.item()
            self.losses['PSNR'] = self.get_psnr(blended_output.detach(), images.detach())
            self.losses['SSIM'] = self.get_ssim(blended_output.detach(), images.detach())
            self.losses['Loss'] = total_loss.item()
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(images)

                bar.suffix = self.print_formatter.format(
                    *[i, len(dataloader)] +
                     [self.losses[key] for key in self.params_monitor]
                )
                bar.next()
                end = time.time()

            if i % 50 == 0:
                # get the whitened image
                masked_out = images.clone()
                masked_out[~occl_gt.expand(-1,3,-1,-1).bool()] = masked_out[~occl_gt.expand(-1,3,-1,-1).bool()]*0.6 + 0.4

                occl_uv[occl_uv >= 0.5] = 1
                occl_uv[occl_uv < 0.5] = 0
                masked_tex_in = tex_gt_uv.clone()
                masked_tex_in[~occl_uv.expand(-1,3,-1,-1).bool()] = masked_tex_in[~occl_uv.expand(-1,3,-1,-1).bool()]*0.6 + 0.4
                tex_in_uv = tex_gt_uv * occl_uv

                self.tb_writer.add_scalar('Test/Recon_Loss', recon_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/Recon_Loss_Image', recon_loss_image.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/SymLoss', symmetry_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/Total_Loss', total_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/batch_time', batch_time, epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/PSNR', self.losses['PSNR'], epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Test/SSIM', self.losses['SSIM'], epoch*len(dataloader)+i)

                self.tb_writer.add_images('Test/Original', normalize(images[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Input', normalize(image_in[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Input_Mask_Overlay', normalize(masked_out[:8]).detach().cpu())
                if self.use_conf:
                    self.tb_writer.add_images('Test/ConfidenceMap', conf_uv[:8].detach().cpu())
                    self.tb_writer.add_images('Test/ConfImage', conf_img[:8].detach().cpu())
                self.tb_writer.add_images('Test/Albedo_In', normalize(alb_in_uv[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Albedo_Gt', normalize(alb_gt_uv[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Albedo_Out', normalize(alb_out_uv[:8]).detach().cpu())
                if self.is_using_symmetry:
                    self.tb_writer.add_images('Test/Albedo_Flip', normalize(alb_out_uv_flip[:8]).detach().cpu())
                    self.tb_writer.add_images('Test/Albedo_Diff', normalize((alb_out_uv - alb_out_uv_flip)[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Occl_OutUv', occl_uv[:8].cpu())
                self.tb_writer.add_images('Test/Recon', normalize(recon_output[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Blended', normalize(blended_output[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Complete Blended', normalize(complete_blended_output[:8]).detach().cpu())
                self.tb_writer.add_images('Test/Gate1_1', gate1[:8,5:6].detach().cpu())
                self.tb_writer.add_images('Test/Gate1_2', gate1[:8,21:22].detach().cpu())
                self.tb_writer.add_images('Test/Gate2_1', gate2[:8,5:6].detach().cpu())
                self.tb_writer.add_images('Test/Gate2_2', gate2[:8,37:38].detach().cpu())

        if self.log_type == 'progressbar':
            bar.finish()
        celeba_flist.close()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        return self.monitor.getvalues('PSNR')+20*self.monitor.getvalues('SSIM')


