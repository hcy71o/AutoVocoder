# Autovocoder: Fast Waveform Generation from a Learned Speech Representation using Differentiable Digital Signal Processing
This repo try to implement [Autovocoder: Fast Waveform Generation from a Learned Speech Representation using Differentiable Digital Signal Processing](https://arxiv.org/abs/2211.06989).
![](AutoVocoder.jpeg)
`Disclaimer : This repo is build for testing purpose. The code is not optimized for performance.`
## Training :
```
python train.py --config config_v1.json
```

## Note:
<!-- * We are able to get good quality of audio with 30 % less training compared to original hifigan.
* This model approx 60 % faster than counterpart hifigan. -->

## Citations :
```
@article{Webber2022AutovocoderFW,
  title={Autovocoder: Fast Waveform Generation from a Learned Speech Representation using Differentiable Digital Signal Processing},
  author={Jacob J. Webber and Cassia Valentini-Botinhao and Evelyn Williams and Gustav Eje Henter and Simon King},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.06989}
}
```

## References:
* https://github.com/jik876/hifi-gan
* https://github.com/rishikksh20/iSTFTNet-pytorch
