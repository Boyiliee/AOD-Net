We appreciate the [PyTorch implementation](https://github.com/TheFairBear/PyTorch-Image-Dehazing) of AOD-Net. Based on this code, we add simple [PONO](https://github.com/Boyiliee/PONO) into AOD-Net, which improves the performance efficiently.

Previous AOD-Net Results:
![](./AOD-Net_result.png)

For TestSet A, the PSNR increases from 19.69 to 20.38 dB, the SSIM increases from 0.8478 to 0.8587. For TestSetB, the PSNR increases from 21.54 to 21.67 dB, the SSIM increases from 0.9272 to 0.9285.

If you find this repo useful, please cite:
```
@inproceedings{ICCV17a,
  title={AOD-Net: All-in-One Dehazing Network},
  author={Li, Boyi and Peng, Xiulian and Wang, Zhangyang and Xu, Ji-Zheng and Feng, Dan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2017}
}

@article{pono,
  title={Positional Normalization},
  author={Li, Boyi and Wu, Felix and Weinberger, Kilian Q. and Belongie, Serge},
  journal={arXiv preprint arXiv:1907.04312},
  year={2019}
}
```
