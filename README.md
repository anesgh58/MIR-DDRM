# MIR-DDRM
**Medical Imaging Restoration via DDRM**

This project utilizes diffusion models to solve inverse problems in an unsupervised manner, building on the DDRM framework: [DDRM](https://github.com/bahjat-kawar/ddrm/tree/master). The degraded image experiences various degradations, including subsampling and blurring, alongside Gaussian noise corruption. The model aims to reconstruct and denoise the image from both observation and model noise by sampling in the SVD space of the degradation matrix. Pretrained models used for this purpose are sourced from [OpenAI - Guided Diffusion](https://github.com/openai/guided-diffusion), trained on the ImageNet dataset.

The novelty of this work lies in applying the model to medical images characterized by BCCB impulse responses, where Singular Value Decomposition (SVD) is replaced by spectral decomposition in the Fourier space.

Refer to the author's original description in "Pretrained models" and "Sampling from the model" for the arborescence of the code and command line execution arguments.

### Example Commands to Reconstruct Blurred Photographic and Ultrasound Images

```bash
python main.py --ni --config imagenet_256.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_imgnet_256_sigma_0
python main.py --ni --config imagenet_512_cc.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_imgnet__512_sigma_0
python main.py --ni --config deblur_us.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_bccb --sigma_0 0 -i deblur_us_sigma_0
```

For a practical demonstration of the deconvolution process applied to photographic images, please refer to the Jupyter notebook located at `/MIR_DDRM.ipynb`. This notebook illustrates the application of the diffusion models used in this project, showcasing an example of the results obtained from the deconvolution process.


## References and Acknowledgements
```
@inproceedings{anesgh58,
    title={Medical Imaging Restoration via DDRM},
    author={Anes Ghouli, Duong Hung PHAM, and Denis KOUAME},
    booktitle={TBD},
    year={2024}
}
```
