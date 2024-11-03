import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from accelerate import Accelerator
import math


from models.diffusion import Model
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.denoising import efficient_generalized_steps

import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model,create_model_2, create_classifier, classifier_defaults, args_to_dict

import random
import scipy.io
# import h5py


def add_AWGN(myimage, SNRdB):
    """
    Add Additive White Gaussian Noise (AWGN) to an image using PyTorch tensors based on the given SNR in dB.

    Parameters:
    - myimage: Input image (as a PyTorch tensor)
    - SNRdB: Signal-to-noise ratio in dB

    Returns:
    - output_image: Image with added AWGN
    """
    # Convert the image to double precision (floating point)
    output_image = myimage.to(torch.float64)

    # Compute the SNR from dB to linear scale
    SNR = 10 ** (SNRdB / 10.0)

    # Calculate the signal power of the image
    ps_in = torch.sum(output_image ** 2) / output_image.numel()

    # Calculate the noise power
    pb_norm = ps_in / SNR

    sigma_0 = torch.sqrt(pb_norm)

    # Generate Gaussian noise
    noise = sigma_0 * torch.randn_like(output_image)

    # Add noise to the original image
    output_image = output_image + noise

    return output_image, sigma_0



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None

        if self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()

            if self.config.data.image_size == 256:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)
            elif self.config.data.image_size == 128:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/128x128_diffusion.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt',
                        ckpt)
            
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/512x512_diffusion.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt',
                        ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
            
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/512x512_classifier.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt',
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)
    
                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale
                cls_fn = cond_fn

        elif self.config.model.type == 'DDPM':
            
            config_dict = vars(self.config.model)

            
            # Convert dim to an integer if it's a string
            if isinstance(config_dict['dim'], str):
                config_dict['dim'] = int(config_dict['dim'].strip(','))

            # Convert dim_mults to a tuple if it's a string
            if isinstance(config_dict['dim_mults'], str):
                config_dict['dim_mults'] = (1, 2, 4, 8)
                # tuple(map(int, config_dict['dim_mults'].strip('()').replace(' ', '').split(',')))

            # Convert image_size to a tuple if it's a string
            if isinstance(config_dict['image_size'], str):
                config_dict['image_size'] = int(config_dict['image_size'].strip('()').strip(','))

            if isinstance(self.config.data.image_size, str):
                self.config.data.image_size = int(self.config.data.image_size.strip('()').strip(','))

            if isinstance(self.config.data.channels, str):
                self.config.data.channels = int(self.config.data.channels.strip(','))

            diffusion = create_model_2(**config_dict)
            ckpt = os.path.join(self.config.model.ckpt_folder,f'{self.config.model.ckpt_name}_{self.config.model.image_size}x{self.config.model.image_size}_diffusion_uncond.pt')
            diffusion.load_state_dict(torch.load(ckpt, map_location=self.device))
            model = diffusion.model # unet model

            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
         

        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config

        #get original images and corrupted y_0
        dataset, test_dataset = get_dataset(args, config)
        
        device_count = torch.cuda.device_count()
        
        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = 1000
            # len(test_dataset)

        # print(f'Dataset has size {len(test_dataset)}')    
        
        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        

        ## get degradation matrix ##
        deg = args.deg
        H_funcs = None
        if deg[:2] == 'cs':
            compress_by = int(deg[2:])
            from functions.svd_replacement import WalshHadamardCS
            H_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by, torch.randperm(self.config.data.image_size**2, device=self.device), self.device)
        elif deg[:3] == 'inp':
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                missing_r = torch.randperm(config.data.image_size**2)[:config.data.image_size**2 // 2].to(self.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'deno':
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride = factor)
        elif deg == 'deblur_uni':
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.device), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_replacement import Deblurring
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            blur_by = 1

            
        elif deg == 'deblur_aniso':
            from functions.svd_replacement import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.device)
            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, self.config.data.image_size, self.device)

        elif deg == 'deblur_bccb':
            from functions.svd_replacement import deconvolution_BCCB
            sigma = 20
            kernel_size = 20
            x_values = torch.linspace(-3 * sigma, 3 * sigma, steps=kernel_size)
            # Compute the 1D Gaussian kernel
            kernel_1d = torch.exp(-0.5 * (x_values / sigma) ** 2)
            kernel_1d = kernel_1d/kernel_1d.sum()
            kernel = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
            H_funcs = deconvolution_BCCB( kernel,self.config.data.image_size, self.device)
            blur_by = 1
        
        elif deg[:2] == 'sr':
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
            
        elif deg == 'color':
            from functions.svd_replacement import Colorization
            H_funcs = Colorization(config.data.image_size, self.device)
        else:
            print("ERROR: degradation type not supported")
            quit()
        args.sigma_0 = 2 * args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = args.sigma_0
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        psnr_list = []
        x0_preds = []
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            # x_orig = x_orig[:, 0, :, :]  
            x_orig = data_transform(self.config, x_orig)

            if self.config.model.degradation:
                y_0 = H_funcs.H(x_orig)
                
                
            else:
                y_0 = x_orig.clone() # already degraded

            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

            #pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            y_0_img = y_0.reshape(y_0.shape[0], config.data.channels, self.config.data.image_size//blur_by, self.config.data.image_size//blur_by)

            if deg[:6] == 'deblur': pinv_y_0 = y_0.view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size)
            elif deg == 'color': pinv_y_0 = y_0.view(y_0.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1, 3, 1, 1)
            elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

            for i in range(len(y_0_img)):
                tvu.save_image(inverse_data_transform(config, torch.real(y_0_img[i])), os.path.join(self.args.image_folder, f"y0_{idx_so_far + i}.png"))

                tvu.save_image(inverse_data_transform(config, x_orig[i]), os.path.join(self.args.image_folder, f"orig_{idx_so_far + i}.png"))

            
            ##Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            
            
            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                x, x0_preds_batch = self.sample_image(x, model, H_funcs, y_0, sigma_0, last=False, cls_fn=cls_fn, classes=classes)

            #x0_preds.append(x0_preds_batch)

            x = [inverse_data_transform(config, y.real.to(dtype=torch.float32)) for y in x]

            folder_path = args.image_folder

            for i in [-1]: #range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{i}.png")
                    )
                    if self.config.model.known_GT:
                        if i == len(x)-1 or i == -1:
                            orig = inverse_data_transform(config, x_orig[j])
                            mse = torch.mean((x[i][j].to(self.device) - orig) ** 2)
                            psnr_list.append(10 * torch.log10(1 / mse))
                            psnr = 20 * torch.log10(torch.tensor(1 / math.sqrt(mse)))
                            avg_psnr += psnr
                            #avg_psnr += psnr[-1]
                            print(psnr)
                            # print(psnr[-1])

            idx_so_far += y_0.shape[0]
            if self.config.model.known_GT:
                pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))
                
        if self.config.model.known_GT:
            psnr_cpu = [p.cpu().numpy() for p in psnr_list]
            scipy.io.savemat(os.path.join(folder_path, 'psnr_values.mat'), {'psnr': psnr_cpu})
            avg_psnr = avg_psnr / (idx_so_far - idx_init)
            print("Total Average PSNR: %.2f" % avg_psnr)

        print("Number of samples: %d" % (idx_so_far - idx_init))

    def sample_image(self, x, model, H_funcs, y_0, sigma_0, last=True, cls_fn=None, classes=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn, classes=classes)
        if last:
            x = x[0][-1]
        return x
