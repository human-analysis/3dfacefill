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

def get_mse(img1, img2):
    img1 = normalize(img1) * 255
    img2 = normalize(img2) * 255
    mse = np.mean((img1 - img2) ** 2)
    return mse

def get_psnr(mse):
    return 20*np.log10(255.0 / np.sqrt(mse))

class Evaluator:
    def __init__(self, args, model, modelO, modelR, renderer, tb_writer=None):
        self.args = args
        self.modelE, self.modelDS, self.modelDT = model[:3]
        self.use_conf = args.use_conf
        self.modelO = modelO
        self.modelR = modelR

        self.save_results = args.save_results
        self.output_dir = './output'
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
        self.reg_resolution = (224, 224)
        self.texture_size = args.texture_size
        self.nchannels = args.nchannels
        self.vertex_num = args.vertex_num
        self.faces = self.renderer.tri[:,:-1].transpose(1,0).cpu().numpy().astype('int32')
        self.is_using_symmetry = args.is_using_symmetry
        self.erosion = morphology.Erosion2d(in_channels=1, out_channels=1, kernel_size=3, soft_max=False, device=self.device)
        self.dilation = morphology.Dilation2d(in_channels=1, out_channels=1, kernel_size=7, soft_max=False, device=self.device)

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'EvalLogger.txt',
            self.save_results
        )
        self.params_loss = ['PSNR', 'SSIM']
        self.log_loss.register(self.params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'PSNR': {'dtype': 'running_mean'},
            'SSIM': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display testing progress
        if self.log_type == 'traditional':
            self.print_formatter = 'Eval [%d/%d][%d/%d] '
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

    def evaluate(self, epoch, dataloader):
        # dataloader = dataloader['test']
        self.monitor.reset()

        # Switch 3DMM models to eval mode
        self.model_eval()
        self.modelR.eval()

        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Eval'), max=len(dataloader))
        end = time.time()

        for i, (images, filenames) in enumerate(dataloader):            # keeps track of data loading time          # Faces
            data_time = time.time() - end

            images = images.to(self.device)
            image_crop = images[:,:,16:240,16:240]
            batch_size = images.shape[0]

            face_mask = torch.zeros_like(images)[:,0:1]
            occl_gt = torch.zeros_like(images)[:,0:1]
            image_mask = torch.zeros_like(images)[:,0:1]
            with torch.no_grad():
                face_parsing_out = self.modelO(image_crop).to(self.device)
                face_mask[:,:,16:240,16:240] = threshold(face_parsing_out[:,0:1])
                occl_gt[:,:,16:240,16:240] = self.dilation(threshold(face_parsing_out[:,1:2]))    # occ_mask
                background_mask = threshold(face_parsing_out[:,2:3])
                image_mask[:,:,16:240,16:240] = 1 - background_mask        # face_mask for occlusion removal
            image_mask_crop = image_mask[:,:,16:240,16:240]

            # # for real occlusion
            # occl_crop = occl_gt[:,:,16:240,16:240]
            # face_mask = threshold(face_mask[:,0:1,16:240,16:240].to(self.device))

            # create mask
            occl_gt = torch.ones_like(images)[:,0:1]
            row_o = torch.randint(images.shape[2]//4, images.shape[2]//2, (batch_size, 1)).squeeze()
            col_o = torch.randint(images.shape[3]//4, images.shape[3]//2, (batch_size, 1)).squeeze()
            height_o = torch.randint(images.shape[2]//3, images.shape[2]*3//4, (batch_size, 1)).squeeze()
            width_o = torch.randint(images.shape[3]//3, images.shape[3]*3//4, (batch_size, 1)).squeeze()
            for j in range(batch_size):
                occl_gt[j, :, row_o[j]:min(row_o[j]+height_o[j], images.shape[2]-1), col_o[j]:min(col_o[j]+width_o[j], images.shape[3]-1)] = 0
            occl_gt = (occl_gt.bool() + (1-image_mask).bool()).float()

            occl_crop = occl_gt[:,:,16:240,16:240]
            occl_crop *= image_mask_crop
            image_in_full = images * (1 - occl_gt)
            image_in = image_crop * (1 - occl_crop)
            occ_percent = (occl_crop.sum(dim=(1,2,3))*10 / image_mask.sum(dim=(1,2,3))).cpu().numpy().astype(int)*10

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
            tex_in_vt, tex_in_vt_mask = self.renderer.get_vt_from_image(image_crop, vertex2d_out, m_normalf_out, m_normal_out)
            vis_in_vt = self.renderer.get_vt_from_image(1-occl_crop, vertex2d_out, m_normalf_out, m_normal_out)[0]
            tex_mask_vt = self.renderer.get_vt_from_image(image_mask, vertex2d_out, m_normal=m_normal_out, m_normalf=m_normalf_out)[0]

            tex_gt_uv, rendering_mask = self.renderer.get_uv_from_vt(tex_in_vt, tex_in_vt_mask)
            vis_uv = threshold(self.renderer.get_uv_from_vt(vis_in_vt, tex_in_vt_mask)[0])
            tex_mask_uv = threshold(self.renderer.get_uv_from_vt(tex_mask_vt, tex_in_vt_mask)[0])

            shade_uv[shade_uv < 0.5] = 0.5
            alb_gt_uv = normalize(tex_gt_uv) / (shade_uv + 1e-8)
            alb_gt_uv_max = alb_gt_uv.max()
            alb_gt_uv -= alb_gt_uv_max/2
            alb_in_uv = alb_gt_uv * vis_uv

            # Run the network
            alb_out_uv, (gate1, gate2, attn), conf_uv = self.modelR(alb_in_uv, mask=vis_uv)

            # Reconstructed output
            tex_out_uv = (2*((alb_out_uv + alb_gt_uv_max/2) * shade_uv - 0.5)).clamp(-1, 1)
            tex_out_vt, tex_out_vt_mask = self.renderer.get_vt_from_uv(tex_out_uv)
            recon_output, recon_output_mask = self.renderer.get_image_from_mapping_params(tex_out_vt, tex_in_vt_mask, verts_map, barycoords, trimap_mask)

            conf_vt, _ = self.renderer.get_vt_from_uv(conf_uv)
            conf_img, _ = self.renderer.get_image_from_mapping_params(conf_vt, tex_in_vt_mask, verts_map, barycoords, trimap_mask)

            # Blended output
            blend_mask = image_mask_crop * recon_output_mask.unsqueeze(1)
            occl_crop = self.erosion(occl_crop * blend_mask).float()
            blended_output = images.clone()
            complete_blended_output = images.clone()

            complete_blended_output[:,:,16:240,16:240] = image_crop * (1 - blend_mask) + recon_output * blend_mask
            blended_output[:,:,16:240,16:240] = image_crop * (1-occl_crop) + recon_output * occl_crop

            self.losses['PSNR'] = self.get_psnr(blended_output.detach(), images.detach())
            self.losses['SSIM'] = self.get_ssim(blended_output.detach(), images.detach())
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(image_crop)

                bar.suffix = self.print_formatter.format(
                    *[processed_data_len, len(dataloader.sampler)] +
                     [self.losses[key] for key in self.params_monitor]
                )
                bar.next()
                end = time.time()

            for j in range(batch_size):
                if occ_percent[j] > 20:
                    # save_path = os.path.join(self.celebamask_path, '{}'.format(i*self.batch_size+j))
                    name_split = filenames[j].split('/')
                    save_path = os.path.join(self.output_dir, name_split[-1].split('.')[0]) # name_split[-2], name_split[-1]
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_image(images[j].detach().cpu(), os.path.join(save_path, 'original.jpg'), normalize=True)
                    save_image(image_in_full[j].detach().cpu(), os.path.join(save_path, 'input.jpg'), normalize=True)
                    save_image(occl_gt[j].detach().cpu(), os.path.join(save_path, 'mask.jpg'))
                    save_image(blended_output[j].detach().cpu(), os.path.join(save_path, 'output.jpg'), normalize=True)
                    save_image(complete_blended_output[j].detach().cpu(), os.path.join(save_path, 'complete_output.jpg'), normalize=True)
                    save_image(tex_out_uv[j].detach().cpu(), os.path.join(save_path, 'texture_out.jpg'), normalize=True)
                    np.save(os.path.join(save_path, 'shape.npy'), shape_out_full[j].view(-1,3).detach().cpu().numpy())
                    with open(os.path.join(save_path, 'occ_percent.txt'), 'w') as occ_file:
                        occ_file.write('{}'.format(occ_percent[j]))

        if self.log_type == 'progressbar':
            bar.finish()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        return self.monitor.getvalues('PSNR')+20*self.monitor.getvalues('SSIM')
