<h1 align="center"> Learning to Handle Complex Constraints for Vehicle Routing Problems </h1>

<p align="center">
<a href="https://neurips.cc/Conferences/2024"><img alt="License" src="https://img.shields.io/static/v1?label=NeurIPS'24&message=Vancouver&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://neurips.cc/virtual/2024/poster/95638"><img alt="License" src="https://img.shields.io/static/v1?label=NeurIPS&message=Poster&color=blue&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href=""><img src="https://img.shields.io/static/v1?label=ArXiv&message=PDF&color=red&style=flat-square" alt="Paper"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href=""><img alt="License" src="https://img.shields.io/static/v1?label=Download&message=Slides&color=orange&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/jieyibi/PIP-constraint/blob/main/LICENSE"><img 
alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=rose&style=flat-square"></a>
</p>

---

Hi there! Thanks for your attention to our work!🤝

This is the PyTorch code for the **Proactive Infeasibility Prevention (PIP)** 
framework implemented on [POMO](https://github.com/yd-kwon/POMO), [AM](https://github.com/wouterkool/attention-learn-to-route) and [GFACS](https://github.com/ai4co/gfacs).

PIP is a generic and effective framework to advance the capabilities of 
neural methods towards more complex VRPs. First, it integrates the Lagrangian 
multiplier as a basis to enhance constraint awareness and introduces 
preventative infeasibility masking to proactively steer the solution 
construction process. Moreover, we present PIP-D, which employs an auxiliary 
decoder and two adaptive strategies to learn and predict these tailored 
masks, potentially enhancing performance while significantly reducing 
computational costs during training. 

For more details, please see our paper [Learning to Handle Complex 
Constraints for Vehicle Routing Problems](), which has been accepted at 
NeurIPS 2024😊. If you find our work useful, please cite:

```
@inproceedings{
    bi2024learning,
    title={Learning to Handle Complex Constraints for Vehicle Routing Problems},
    author={Bi, Jieyi and Ma, Yining and Zhou, Jianan and Song, Wen and Cao, 
    Zhiguang and Wu, Yaoxin and Zhang, Jie},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2024}
}
```
---

## Acknowledgments
https://github.com/yd-kwon/POMO  
https://github.com/wouterkool/attention-learn-to-route  
https://github.com/ai4co/gfacs  
https://github.com/RoyalSkye/Routing-MVMoE
