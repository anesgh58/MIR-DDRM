# MIR-DDRM
**Medical Imaging Restoration via DDRM**

This project utilizes diffusion models to address inverse problems in an unsupervised manner. Implemmentation is based on DDRM paper: ([DDRM](https://github.com/bahjat-kawar/ddrm/tree/master)). The degraded image undergoes several types of degradation, including subsampling and blurring, and is corrupted by Gaussian noise. The model aims to reconstruct the image while simultaneously denoising it from both observation noise and model noise. A single pretrained model is used for this purpose, with pretrained models sourced from: [Open AI - guided-diffusion](https://github.com/openai/guided-diffusion). The models are trained on the ImageNet dataset.

The contribution and novelty of this work lie in the application of the model to medical images characterized by BCCB impulse responses. In this context, Singular Value Decomposition (SVD) is replaced by spectral decomposition in the Fourier space.

Refer to the author's original description in "Sampling from the model" for the different arguments used in the command line execution.

### Example Commands to Reconstruct Blurred Photographic and Ultrasound Images

```bash
python main.py --ni --config imagenet_256.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_imgnet_256_sigma_0
python main.py --ni --config imagenet_512_cc.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_imgnet__512_sigma_0
python main.py --ni --config deblur_us.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_us_sigma_0
```

## References and Acknowledgements

@inproceedings{anesgh58,
    title={Medical Imaging Restoration via DDRM},
    author={Anes Ghouli, Duong Hung PHAM, and Denis KOUAME},
    booktitle={TBD},
    year={2024}
}
