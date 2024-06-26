# ProMo
A Bayesian Probabilistic Framework for Mechanical Fault Diagnosis. The codes for ["Integrating Misidentification and OOD Detection for Reliable Fault Diagnosis of High-Speed Train Bogie"](https://ieeexplore.ieee.org/document/10480918).

## Abstract
<p align="justify">
Building a credible fault diagnosis framework, for critical mechanical components such as high-speed train bogie, faces two key challenges, precisely identifying unknown faults and confidently predicting known faults. Despite recent initial advances in reliable fault diagnosis, out-of-distribution (OOD) detection with covariate shift has not been studied, and the aforementioned two core issues are often treated separately. In this paper, a Probabilistic framework for Mechanical fault diagnosis (ProMo) is proposed to effectively integrate misidentification and OOD detection. The core idea is to use model parameter distributions to capture uncertainties of the input and the model, so the misclassified and OOD samples with high uncertainty can be separated. Specifically, ProMo is a Bayesian deep network, where hierarchical predictions are for robust uncertainty estimation, and probabilistic null space analysis of model weights is for accurate OOD detection. The hierarchical structure decomposes model uncertainty into different depths while providing multiple interfaces for diagnostic results. To enhance OOD detection ability, model weights are probabilistically projected into null space to calculate OOD score. Extensive experiments demonstrated that ProMo outperforms state-of-the-art methods and achieves reliable fault diagnosis, even in case with covariate shift on fault severity and working condition. 
</p>

<div align=center>
<img src="figs/framework_illustration_v2.png" width="800">
</div>
<p align="justify">
Fig. 1. Overview of the proposed reliable probabilistic fault diagnosis framework. The 1st row shows the pipeline of the method, and the 2nd presents the two main innovations of this article. The InD and OOD data collected from different devices are input into a Bayesian model to obtain the basic features, the predictions are generated through hierarchical classifiers and Monte Carlo estimation (not shown), and finally reliable diagnosis and OOD detection results are obtained through uncertainty estimation and probability null space analysis.
</p>

## Proposed Method
<p align="justify">
The backbone of the proposed approach is a fully Bayesian deep network. As depicted in Fig. 2, the proposed method integrates misidentification and OOD detection in a unified end-to-end framework, in which two tools are given (PNuSA score and MC uncertainty). Uncertainty estimation and OOD scores evaluate the model uncertainty from different aspects, and do not serve as training objectives and thus can be plugand-play. The core challenge is how to construct a Bayesian model sensitive to OOD samples and meanwhile retain good identification ability. In ProMo, a hierarchical structure is presented, uncertainty can be collected from multi-level classifiers, OOD score is computed via weight distributions of classifiers, and prediction accuracy is significantly enhanced via MC estimation over the proposed probabilistic model.
</p>

The detailed structure is put in ``models.py``. You could modify it to run on your constructed dataset.

## Citation
If you use our codes in your work, please cite it as
```
@article{feng2024integrating,
  author={Feng, Yong and Chen, Jinglong and Xie, Zongliang and Xie, Jingsong and Pan, Tongyang and Li, Chao and Zhao, Qing},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Integrating Misidentification and OOD Detection for Reliable Fault Diagnosis of High-Speed Train Bogie}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  keywords={Uncertainty;Reliability;Fault diagnosis;Bayes methods;Probabilistic logic;Predictive models;Data models;Fault diagnosis;high-speed train bogie;Bayesian learning;out-of-distribution detection;uncertainty estimation},
  doi={10.1109/TITS.2024.3377813}}
```
