# train.py

import time
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import evaluate
from torchvision.utils import save_image
import plugins
import numpy as np
from rendering import *
from models import UnetSeg, VGG_16, UnetSelfAttnConf32, Discriminator_localglobal
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import euler_angles_to_matrix
from checkpoints import Checkpoints
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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif hasattr(m, 'weight_bar'):
            m.weight_bar.data.normal_(0, math.sqrt(2. / n))
            m.weight_u.data.normal_(0, math.sqrt(2. / n))
            m.weight_v.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    else:
        pass

class Trainer:
    def __init__(self, args, model, modelO, criterion):
        self.args = args
        self.modelE, self.modelDS, self.modelDT = model[:3]
        self.use_conf = args.use_conf
        self.modelO = modelO
        self.criterion = criterion

        self.save_results = args.save_results
        self.checkpoint = Checkpoints(args)
        args.resolution = (224, 224)
        self.renderer = Rendering(args)
        self.resume = args.resume

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

        self.is_using_symmetry = args.is_using_symmetry
        self.is_using_frecon = args.is_using_frecon
        self.dilation = morphology.Dilation2d(in_channels=1, out_channels=1, kernel_size=3, soft_max=False, device=self.device)

        # load 3DMM model
        _3dmm_model_keys = ['modelE', 'modelDT', 'modelDC']
        self._3dmm_ckpt = args._3dmm_ckpt
        if self._3dmm_ckpt is not None:
            checkpoint = self.checkpoint.load_ckpt(filename=self._3dmm_ckpt)
            for key in _3dmm_model_keys:
                if key in checkpoint.keys() and hasattr(self, key):
                    model = getattr(self, key)
                    self.checkpoint.load_state_dict(model, checkpoint[key])

        # load occlusion model
        self.occ_ckpt = args.occ_ckpt
        if self.occ_ckpt is not None:
            checkpoint = self.checkpoint.load_ckpt(filename=self.occ_ckpt)
            self.checkpoint.load_state_dict(self.modelO, checkpoint['modelO']) #

        if self.is_using_frecon:
            self.vgg_face = VGG_16(ngpu=self.ngpu)
            self.vgg_face.load_weights(path='./checkpoints/vgg_face_torch/VGG_FACE.t7')
            self.vgg_face.to(self.devices[-1])
            self.vgg_face = nn.DataParallel(self.vgg_face, device_ids=list(reversed(range(self.ngpu))))#[:-1])
            self.vgg_face.eval()
        else:
            self.vgg_face = None

        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.texture_size = args.texture_size
        self.nchannels = args.nchannels
        self.vertex_num = args.vertex_num
        self.faces = self.renderer.tri[:,:-1].transpose(1,0).cpu().numpy().astype('int32')

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method
        self.scheduler_options = args.scheduler_options

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            self.save_results
        )
        self.params_loss = ['ReconL2', 'SymLoss', 'ReconIL2', 'Loss', 'LossD', 'LossG', 'PercLoss', 'PSNR', 'SSIM']
        self.log_loss.register(self.params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'ReconL2': {'dtype': 'running_mean'},
            'SymLoss': {'dtype': 'running_mean'},
            'Loss': {'dtype': 'running_mean'},
            'LossD': {'dtype': 'running_mean'},
            'LossG': {'dtype': 'running_mean'},
            'PercLoss': {'dtype': 'running_mean'},
            'ReconIL2': {'dtype': 'running_mean'},
            'PSNR': {'dtype': 'running_mean'},
            'SSIM': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display training progress
        if self.log_type == 'traditional':
            self.print_formatter = 'Train [%d/%d][%d/%d] '
            for item in self.params_loss:
                self.print_formatter += item + " %.3f "
        elif self.log_type == 'progressbar':
            self.print_formatter = '({}/{})'
            for item in self.params_loss:
                self.print_formatter += ' | ' + item + ' {:.2f}'
            self.print_formatter += ' | lr: {:.0e}'

        self.tb_writer = SummaryWriter(log_dir=args.logs_dir)
        self.losses = {}
        self.setupParaStat()
        self.setupReconstructionModel()
        self.gan_criterion = nn.BCELoss()
        self.d_iter = 1

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

    ## Work with the mean shape to get Mesh Transforms
    def setupReconstructionModel(self):
        self.no_sym = False
        # Initialize the reconstruction model
        print("Generating Reconstruction Model...\n")
        self.modelR = UnetSelfAttnConf32(in_channels=4, out_channels=3, dropout_p=0.3, bilinear=True, use_attn=False).to(self.device)
        self.modelR.apply(weights_init)
        disc_inc = 3

        print("Generating Image Discriminator...\n")
        self.discI = Discriminator_localglobal(self.args, in_c=disc_inc).to(self.device)
        self.discI.apply(weights_init)

        if self.ngpu > 1:
            self.modelR = nn.DataParallel(self.modelR, list(range(self.ngpu)))
            self.discI = nn.DataParallel(self.discI, list(range(self.ngpu)))#[2:] + [0,1])

        # Optimizer, scheduler and losses
        self.optimizerR = getattr(optim, self.optim_method)(filter(lambda p: p.requires_grad, self.modelR.parameters()), lr=self.lr, **self.optim_options)
        self.schedulerR = getattr(optim.lr_scheduler, self.scheduler_method)(self.optimizerR, **self.scheduler_options)
        self.optimizerDI = getattr(optim, self.optim_method)(filter(lambda p: p.requires_grad, self.discI.parameters()), lr=5*self.lr, **self.optim_options)
        self.schedulerDI = getattr(optim.lr_scheduler, self.scheduler_method)(self.optimizerDI, **self.scheduler_options)

        # Resume from checkpoint if given
        self.epoch = 0
        self.model_keys = ['modelR', 'discI']
        self.optimizer_keys = ['optimizerR', 'optimizerDI']
        self.ckpt_dict = {}
        if self.resume is not None:
            checkpoint = self.checkpoint.load_ckpt()
            for key in self.model_keys:
                if key in checkpoint.keys() and hasattr(self, key):
                    model = getattr(self, key)
                    self.checkpoint.load_state_dict(model, checkpoint[key])
            for key in self.optimizer_keys:
                if key in checkpoint.keys() and hasattr(self, key):
                    optimizer = getattr(self, key)
                    optimizer.load_state_dict(checkpoint[key])
                    if args.update_lr:
                        for param in optimizer.param_groups: param['lr'] = self.lr
            if 'epoch' in checkpoint.keys():
                self.epoch = checkpoint['epoch'] + 1
            self.ckpt_dict = checkpoint

    def calc_gradient_penalty(self, real_data, fake_data, disc, batch_size, mask=None):
        alpha = torch.rand(batch_size, 1)
        while len(alpha.size()) < len(real_data.size()):
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)
        interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = disc(interpolates)

        if type(disc_interpolates) is list:
            gradients = 0
            for j in range(len(disc_interpolates)):
                gradients += autograd.grad(outputs=disc_interpolates[j], inputs=interpolates, grad_outputs=torch.ones(disc_interpolates[j].size()).to(real_data.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
        if mask is not None:
            mask = mask.to(gradients.device)
            gradient_penalty *= (mask * (np.prod([*mask.shape]) / mask.sum()))

        return gradient_penalty.mean()

    def model_eval(self):
        self.modelE.eval()
        self.modelDS.eval()
        self.modelDT.eval()
        if self.modelO is not None:
            self.modelO.eval()

    def model_train(self):
        self.modelR.train()
        self.discI.train()

    def train(self, epoch, dataloader):
        dataloader = dataloader['train']
        self.monitor.reset()

        # Switch 3DMM models to eval mode
        self.model_eval()
        self.model_train()

        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Train'), max=len(dataloader))
        end = time.time()

        gen_loss = torch.zeros([], device=self.device)
        disc_loss = torch.zeros([], device=self.device)
        gp_loss = torch.zeros([], device=self.device)
        d_iter=0
        for i, (images, filenames) in enumerate(dataloader):                                                        # Faces

            # Clean the model gradrients
            self.optimizerR.zero_grad()
            self.optimizerDI.zero_grad()

            # keeps track of data loading time
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
            row_o = torch.randint(images.shape[2]//4, images.shape[2]//2, (batch_size,))
            col_o = torch.randint(images.shape[3]//4, images.shape[3]//2, (batch_size,))
            height_o = torch.randint(images.shape[2]//4, images.shape[2]*3//4, (batch_size,))
            width_o = torch.randint(images.shape[3]//4, images.shape[3]*3//4, (batch_size,))
            for j in range(batch_size):
                occl_gt[j, :, row_o[j]:min(row_o[j]+height_o[j], images.shape[2]-1), col_o[j]:min(col_o[j]+width_o[j], images.shape[3]-1)] = 0
            occl_gt = (occl_gt.bool() + (1-image_mask).bool()).float()
            image_in = images * occl_gt

            ############################
            # Run the network
            ############################
            # get 3DMM fit
            with torch.no_grad():
                shape_feats, tex_feats, conf_feats, scale_out, pose_out, il_feats = self.modelE(images)   # images
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

            # Extract the texture of the input and original images and mask using the obtained shape
            tex_in_vt, tex_in_vt_mask = self.renderer.get_vt_from_image(images, vertex2d_out, m_normalf_out, m_normal_out)
            occ_in_vt = self.renderer.get_vt_from_image(occl_gt, vertex2d_out, m_normalf_out, m_normal_out)[0]
            tex_mask_vt = self.renderer.get_vt_from_image(image_mask, vertex2d_out, m_normal=m_normal_out, m_normalf=m_normalf_out)[0]

            tex_gt_uv, rendering_mask = self.renderer.get_uv_from_vt(tex_in_vt, tex_in_vt_mask)
            occl_uv = threshold(self.renderer.get_uv_from_vt(occ_in_vt, tex_in_vt_mask)[0])
            tex_mask_uv = threshold(self.renderer.get_uv_from_vt(tex_mask_vt, tex_in_vt_mask)[0])
            occl_uv *= tex_mask_uv

            # remove shade from texture and obtain albedo in the UV domain
            shade_uv[shade_uv < 0.5] = 0.5
            alb_gt_uv = normalize(tex_gt_uv) / (shade_uv + 1e-8)
            alb_gt_uv_max = alb_gt_uv.max()
            alb_gt_uv -= alb_gt_uv_max/2
            alb_in_uv = alb_gt_uv * occl_uv

            # Run the albedo completion network
            alb_out_uv, (gate1, gate2, attn), conf_uv = self.modelR(alb_in_uv, mask=occl_uv)

            # compute losses
            recon_loss = self.criterion.norm_loss(alb_out_uv, alb_gt_uv, mask=tex_mask_uv, conf=conf_uv, loss_type='l1')
            factor = recon_loss.item() / 50
            if recon_loss.item() > 50:
                recon_loss /= factor

            # Symmetry loss
            if self.is_using_symmetry:
                alb_out_uv_flip = alb_out_uv.flip(dims=(3,)).detach()
                alb_diff_ignore = torch.abs(alb_out_uv - alb_out_uv_flip) <= 0.005
                alb_out_uv_flip[alb_diff_ignore] = alb_out_uv[alb_diff_ignore]
                symmetry_loss = self.criterion.norm_loss(alb_out_uv, alb_out_uv_flip, mask=(1-occl_uv)*occl_uv.flip(dims=(3,)), conf=conf_uv, loss_type='l1')
            factor = symmetry_loss.item() / 50
            if symmetry_loss.item() > 50:
                symmetry_loss /= factor

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
            recon_loss_image = self.criterion.norm_loss(complete_blended_output, images, mask=image_mask, conf=conf_img, loss_type='l1')*1
            factor = recon_loss_image.item() / 50
            if recon_loss_image.item() > 50:
                recon_loss_image /= factor

            perceptual_loss = torch.zeros([], device=self.devices[-1])
            perceptual_multipliers = [1,1,1,1]#[1e-5, 1e-6, 1e-6, 1e-4, 0.1]
            if self.is_using_frecon and epoch >= 0:
                vgg_in_feat = self.vgg_face(images.to(self.devices[-1]))  #(images * blend_mask)
                vgg_out_feat = self.vgg_face(complete_blended_output.to(self.devices[-1])) #(rendered_images * blend_mask)
                for f_idx in range(len(vgg_in_feat)):
                    p_mask = nn.functional.interpolate(blend_mask, scale_factor=1/(2**(f_idx))).to(self.devices[-1])
                    conf_scaled = F.interpolate(conf_img, scale_factor=vgg_out_feat[f_idx].shape[-1]/conf_img.shape[-1]).to(self.devices[-1]).detach()
                    p_loss = self.criterion.norm_loss(vgg_out_feat[f_idx], vgg_in_feat[f_idx].detach(), mask=p_mask, conf=conf_scaled, loss_type='l1')
                    perceptual_loss += p_loss * perceptual_multipliers[f_idx]
            perceptual_loss = perceptual_loss.to(self.device)   #*5e-0  #*5e-2
            factor = perceptual_loss.item() / 5
            if perceptual_loss.item() > 5:
                perceptual_loss /= factor

            total_loss = recon_loss + symmetry_loss + recon_loss_image
            update_loss = total_loss + perceptual_loss

            # Image Discriminator
            mask = torch.ones(batch_size,1,1,1).to(self.device)
            conf_scaled = None

            if d_iter < self.d_iter:
                # Update discriminator
                real_data = images
                gen_data = blended_output.detach() #complete_blended_output
                scores_real = self.discI(real_data)
                scores_gen = self.discI(gen_data)
                disc_loss = 0
                for j in range(len(scores_real)):
                    conf_scaled = F.interpolate(conf_img, scale_factor=scores_real[j].shape[-1]/conf_img.shape[-1]).to(self.device).detach()
                    disc_loss += self.criterion.gan_loss(nn.functional.relu(1 - scores_real[j]) + nn.functional.relu(1 + scores_gen[j]), mask, conf=conf_scaled)
                disc_loss = disc_loss.to(self.device) / len(scores_real)
                gp_loss = self.calc_gradient_penalty(real_data=real_data, fake_data=gen_data, disc=self.discI, batch_size=batch_size, mask=blend_mask[:,0]).to(self.device)

                factor = disc_loss.abs().item() / 20
                if disc_loss.abs().item() > 20:
                    disc_loss /= factor
                update_loss += disc_loss + 10*gp_loss
            else:
                # Update generator
                gen_data = blended_output.to(self.device)
                scores_gen = self.discI(gen_data)
                gen_loss = 0
                for j in range(len(scores_gen)):
                    conf_scaled = F.interpolate(conf_img, scale_factor=scores_gen[j].shape[-1]/conf_img.shape[-1]).to(self.device).detach()
                    gen_loss -= self.criterion.gan_loss(scores_gen[j], mask, conf=conf_scaled)
                gen_loss = gen_loss.to(self.device) / len(scores_gen)

                factor = gen_loss.abs().item() / 20
                if gen_loss.abs().item() > 20:
                    gen_loss /= factor
                update_loss += gen_loss

            update_loss.backward()
            self.optimizerR.step()

            if d_iter < self.d_iter:
                self.optimizerDI.step()
                d_iter += 1
            else:
                d_iter = 0

            self.losses['ReconL2'] = recon_loss.item()
            self.losses['ReconIL2'] = recon_loss_image.item()
            self.losses['SymLoss'] = symmetry_loss.item()
            self.losses['Loss'] = total_loss.item()
            self.losses['PercLoss'] = perceptual_loss.item()
            self.losses['LossD'] = disc_loss.item()
            self.losses['LossG'] = gen_loss.item()
            self.losses['PSNR'] = self.get_psnr(blended_output.detach(), images.detach())
            self.losses['SSIM'] = self.get_ssim(blended_output.detach(), images.detach())
            self.monitor.update(self.losses, batch_size)

            if self.log_type == 'traditional':
                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i, len(dataloader)] +
                    [self.losses[key] for key in self.params_loss]))
            elif self.log_type == 'progressbar':
                # update progress bar
                batch_time = time.time() - end
                processed_data_len += len(images)

                bar.suffix = self.print_formatter.format(
                    *[i, len(dataloader)] +
                     [self.losses[key] for key in self.params_loss] +
                     [self.optimizerR.param_groups[-1]['lr']]
                )
                bar.next()
                end = time.time()

            if (epoch - self.epoch < 2 and i % 50 == 0) or (i % 100 == 0):
                # get the whitened image
                masked_out = images.clone()#(images * weight_mask).clone()
                masked_out[~occl_gt.expand(-1,3,-1,-1).bool()] = masked_out[~occl_gt.expand(-1,3,-1,-1).bool()]*0.6 + 0.4

                occl_uv[occl_uv >= 0.5] = 1
                occl_uv[occl_uv < 0.5] = 0
                masked_tex_in = tex_gt_uv.clone()
                masked_tex_in[~occl_uv.expand(-1,3,-1,-1).bool()] = masked_tex_in[~occl_uv.expand(-1,3,-1,-1).bool()]*0.6 + 0.4
                tex_in_uv = tex_gt_uv * occl_uv

                self.tb_writer.add_scalar('Train/Recon_Loss', recon_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/Recon_Loss_Image', recon_loss_image.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/SymLoss', symmetry_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/DiscLoss', disc_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/GenLoss', gen_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/GPLoss', gp_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/PercLoss', perceptual_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/PSNR', self.losses['PSNR'], epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/SSIM', self.losses['SSIM'], epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/Total_Loss', total_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/Update_Loss', update_loss.item(), epoch*len(dataloader)+i)
                self.tb_writer.add_scalar('Train/batch_time', batch_time, epoch*len(dataloader)+i)

                self.tb_writer.add_images('Train/Original', normalize(images[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Input', normalize(image_in[:8]).detach().cpu())
                self.tb_writer.add_images('Train/FaceMask', normalize(image_mask[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Input_Mask_Overlay', normalize(masked_out[:8]).detach().cpu())
                if self.use_conf:
                    self.tb_writer.add_images('Train/ConfidenceMap', conf_uv[:8].detach().cpu())
                    self.tb_writer.add_images('Train/ConfImage', conf_img[:8].detach().cpu())

                self.tb_writer.add_images('Train/Albedo_In', normalize(alb_in_uv[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Albedo_Gt', normalize(alb_gt_uv[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Albedo_Out', normalize(alb_out_uv[:8]).detach().cpu())
                if self.is_using_symmetry and not self.no_sym:
                    self.tb_writer.add_images('Train/Albedo_Flip', normalize(alb_out_uv_flip[:8]).detach().cpu())
                    self.tb_writer.add_images('Train/Albedo_Diff', normalize((alb_out_uv - alb_out_uv_flip)[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Occl_OutUv', occl_uv[:8].cpu())
                self.tb_writer.add_images('Train/Recon', normalize(recon_output[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Blended', normalize(blended_output[:8]).detach().cpu())
                self.tb_writer.add_images('Train/Complete Blended', normalize(complete_blended_output[:8]).detach().cpu())

                self.tb_writer.add_images('Train/Gate1_1', gate1[:8,5:6].detach().cpu())
                self.tb_writer.add_images('Train/Gate1_2', gate1[:8,21:22].detach().cpu())
                self.tb_writer.add_images('Train/Gate2_1', gate2[:8,5:6].detach().cpu())
                self.tb_writer.add_images('Train/Gate2_2', gate2[:8,37:38].detach().cpu())

        if self.optimizerR.param_groups[-1]['lr'] > 1e-7:
            self.schedulerR.step()

        if self.optimizerDI.param_groups[-1]['lr'] > 1e-7:
            self.schedulerDI.step()

        # gather current checkpoint
        for key in [*self.model_keys, *self.optimizer_keys]:
            if hasattr(self, key):
                self.ckpt_dict[key] = getattr(self, key).state_dict()
        self.ckpt_dict['epoch'] = epoch

        if self.log_type == 'progressbar':
            bar.finish()
        celeba_flist.close()

        loss = self.monitor.getvalues()
        self.log_loss.update(loss)
        return self.monitor.getvalues('Loss')


