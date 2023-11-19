# ProMo
A Bayesian Probabilistic Framework for Mechanical Fault Diagnosis

## Abstract
Building a credible fault diagnosis framework, for critical mechanical components such as high-speed train bogie, faces two key challenges, precisely identifying unknown faults and confidently predicting known faults. Despite recent initial advances in reliable fault diagnosis, out-of-distribution (OOD) detection with covariate shift has not been studied, and the aforementioned two core issues are often treated separately. In this paper, a Probabilistic framework for Mechanical fault diagnosis (ProMo) is proposed to effectively integrate misidentification and OOD detection. The core idea is to use model parameter distributions to capture uncertainties of the input and the model, so the misclassified and OOD samples with high uncertainty can be separated. Specifically, ProMo is a Bayesian deep network, where hierarchical predictions are for robust uncertainty estimation, and probabilistic null space analysis of model weights is for accurate OOD detection. The hierarchical structure decomposes model uncertainty into different depths while providing multiple interfaces for diagnostic results. To enhance OOD detection ability, model weights are probabilistically projected into null space to calculate OOD score. Extensive experiments demonstrated that ProMo outperforms state-of-the-art methods and achieves reliable fault diagnosis, even in case with covariate shift on fault severity and working condition. 


## Structure

