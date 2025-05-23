# This file is auto-generated. Do not edit manually.
from typing import Dict, Type, TypeVar
from .modules.hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode
from .modules.pytorch_wrapper.pt_abs import PtAbs
from .modules.pytorch_wrapper.pt_acos import PtAcos
from .modules.pytorch_wrapper.pt_add import PtAdd
from .modules.pytorch_wrapper.pt_apply_function import PtApplyFunction
from .modules.pytorch_wrapper.pt_arange import PtArange
from .modules.pytorch_wrapper.pt_argmax import PtArgmax
from .modules.pytorch_wrapper.pt_argmin import PtArgmin
from .modules.pytorch_wrapper.pt_asin import PtAsin
from .modules.pytorch_wrapper.pt_atan import PtAtan
from .modules.pytorch_wrapper.pt_bitwise_and import PtBitwiseAnd
from .modules.pytorch_wrapper.pt_bitwise_left_shift import PtBitwiseLeftShift
from .modules.pytorch_wrapper.pt_bitwise_not import PtBitwiseNot
from .modules.pytorch_wrapper.pt_bitwise_or import PtBitwiseOr
from .modules.pytorch_wrapper.pt_bitwise_right_shift import PtBitwiseRightShift
from .modules.pytorch_wrapper.pt_bitwise_xor import PtBitwiseXor
from .modules.pytorch_wrapper.pt_bmm import PtBmm
from .modules.pytorch_wrapper.pt_bool_create import PtBoolCreate
from .modules.pytorch_wrapper.pt_compute_loss import PtComputeLoss
from .modules.pytorch_wrapper.pt_concat import PtConcat
from .modules.pytorch_wrapper.pt_cos import PtCos
from .modules.pytorch_wrapper.pt_cosh import PtCosh
from .modules.pytorch_wrapper.pt_crop import PtCrop
from .modules.pytorch_wrapper.pt_data_loader import PtDataLoader
from .modules.pytorch_wrapper.pt_data_loader_from_tensors import PtDataLoaderFromTensors
from .modules.pytorch_wrapper.pt_div import PtDiv
from .modules.pytorch_wrapper.pt_einsum import PtEinsum
from .modules.pytorch_wrapper.pt_eq import PtEq
from .modules.pytorch_wrapper.pt_evaluate_classification_model import PtEvaluateClassificationModel
from .modules.pytorch_wrapper.pt_exp import PtExp
from .modules.pytorch_wrapper.pt_flatten import PtFlatten
from .modules.pytorch_wrapper.pt_float_create import PtFloatCreate
from .modules.pytorch_wrapper.pt_floor_divide import PtFloorDiv
from .modules.pytorch_wrapper.pt_from_image import PtFromImage
from .modules.pytorch_wrapper.pt_from_image_transpose import PtFromImageTranspose
from .modules.pytorch_wrapper.pt_from_latent import PtFromLatent
from .modules.pytorch_wrapper.pt_from_numpy import PtFromNumpy
from .modules.pytorch_wrapper.pt_full import PtFull
from .modules.pytorch_wrapper.pt_gather import PtGather
from .modules.pytorch_wrapper.pt_ge import PtGe
from .modules.pytorch_wrapper.pt_gt import PtGt
from .modules.pytorch_wrapper.pt_index_select import PtIndexSelect
from .modules.pytorch_wrapper.pt_int_create import PtIntCreate
from .modules.pytorch_wrapper.pt_interpolate_by_scale_factor import PtInterpolateByScaleFactor
from .modules.pytorch_wrapper.pt_interpolate_to_size import PtInterpolateToSize
from .modules.pytorch_wrapper.pt_le import PtLe
from .modules.pytorch_wrapper.pt_linspace import PtLinspace
from .modules.pytorch_wrapper.pt_load_model import PtLoadModel
from .modules.pytorch_wrapper.pt_log import PtLog
from .modules.pytorch_wrapper.pt_logical_and import PtLogicalAnd
from .modules.pytorch_wrapper.pt_logical_not import PtLogicalNot
from .modules.pytorch_wrapper.pt_logical_or import PtLogicalOr
from .modules.pytorch_wrapper.pt_logical_xor import PtLogicalXor
from .modules.pytorch_wrapper.pt_lt import PtLt
from .modules.pytorch_wrapper.pt_masked_select import PtMaskedSelect
from .modules.pytorch_wrapper.pt_matmul import PtMatMul
from .modules.pytorch_wrapper.pt_max import PtMax
from .modules.pytorch_wrapper.pt_mean import PtMean
from .modules.pytorch_wrapper.pt_median import PtMedian
from .modules.pytorch_wrapper.pt_min import PtMin
from .modules.pytorch_wrapper.pt_mm import PtMm
from .modules.pytorch_wrapper.pt_mul import PtMul
from .modules.pytorch_wrapper.pt_ne import PtNe
from .modules.pytorch_wrapper.pt_neg import PtNeg
from .modules.pytorch_wrapper.pt_ones import PtOnes
from .modules.pytorch_wrapper.pt_pad import PtPad
from .modules.pytorch_wrapper.pt_permute import PtPermute
from .modules.pytorch_wrapper.pt_pow import PtPow
from .modules.pytorch_wrapper.pt_predict_classification_model import PtPredictClassificationModel
from .modules.pytorch_wrapper.pt_predict_regression_model import PtPredictRegressionModel
from .modules.pytorch_wrapper.pt_prod import PtProd
from .modules.pytorch_wrapper.pt_rand import PtRand
from .modules.pytorch_wrapper.pt_rand_int import PtRandInt
from .modules.pytorch_wrapper.pt_randn import PtRandn
from .modules.pytorch_wrapper.pt_remainder import PtRemainder
from .modules.pytorch_wrapper.pt_reshape import PtReshape
from .modules.pytorch_wrapper.pt_save_model import PtSaveModel
from .modules.pytorch_wrapper.pt_scatter import PtScatter
from .modules.pytorch_wrapper.pt_show_size import PtShowSize
from .modules.pytorch_wrapper.pt_show_text import PtShowText
from .modules.pytorch_wrapper.pt_sin import PtSin
from .modules.pytorch_wrapper.pt_sinh import PtSinh
from .modules.pytorch_wrapper.pt_size import PtSize
from .modules.pytorch_wrapper.pt_size_create import PtSizeCreate
from .modules.pytorch_wrapper.pt_size_to_numpy import PtSizeToNumpy
from .modules.pytorch_wrapper.pt_size_to_string import PtSizeToString
from .modules.pytorch_wrapper.pt_sqrt import PtSqrt
from .modules.pytorch_wrapper.pt_squeeze import PtSqueeze
from .modules.pytorch_wrapper.pt_stack import PtStack
from .modules.pytorch_wrapper.pt_std import PtStd
from .modules.pytorch_wrapper.pt_sub import PtSub
from .modules.pytorch_wrapper.pt_sum import PtSum
from .modules.pytorch_wrapper.pt_tan import PtTan
from .modules.pytorch_wrapper.pt_tanh import PtTanh
from .modules.pytorch_wrapper.pt_to_bfloat16 import PtToBfloat16
from .modules.pytorch_wrapper.pt_to_float16 import PtToFloat16
from .modules.pytorch_wrapper.pt_to_float32 import PtToFloat32
from .modules.pytorch_wrapper.pt_to_float64 import PtToFloat64
from .modules.pytorch_wrapper.pt_to_image import PtToImage
from .modules.pytorch_wrapper.pt_to_image_transpose import PtToImageTranspose
from .modules.pytorch_wrapper.pt_to_int16 import PtToInt16
from .modules.pytorch_wrapper.pt_to_int32 import PtToInt32
from .modules.pytorch_wrapper.pt_to_int64 import PtToInt64
from .modules.pytorch_wrapper.pt_to_int8 import PtToInt8
from .modules.pytorch_wrapper.pt_to_latent import PtToLatent
from .modules.pytorch_wrapper.pt_to_numpy import PtToNumpy
from .modules.pytorch_wrapper.pt_to_rgb_tensors import PtToRgbTensors
from .modules.pytorch_wrapper.pt_to_uint8 import PtToUint8
from .modules.pytorch_wrapper.pt_tokenizer import PtTokenizer
from .modules.pytorch_wrapper.pt_train_classification_model import PtTrainClassificationModel
from .modules.pytorch_wrapper.pt_train_classification_model_lr import PtTrainClassificationModelLr
from .modules.pytorch_wrapper.pt_train_classification_transformer_model import PtTrainClassificationTransformerModel
from .modules.pytorch_wrapper.pt_train_fine_tune_classification_transformer_model import PtTrainFineTuneClassificationTransformerModel
from .modules.pytorch_wrapper.pt_train_model import PtTrainModel
from .modules.pytorch_wrapper.pt_train_regression_model import PtTrainRegressionModel
from .modules.pytorch_wrapper.pt_train_rnn_model import PtTrainRNNModel
from .modules.pytorch_wrapper.pt_unsqueeze import PtUnsqueeze
from .modules.pytorch_wrapper.pt_var import PtVar
from .modules.pytorch_wrapper.pt_view import PtView
from .modules.pytorch_wrapper.pt_where import PtWhere
from .modules.pytorch_wrapper.pt_zeros import PtZeros
from .modules.pytorch_wrapper.ptd_bernoulli import PtdBernoulli
from .modules.pytorch_wrapper.ptd_beta import PtdBeta
from .modules.pytorch_wrapper.ptd_binomial import PtdBinomial
from .modules.pytorch_wrapper.ptd_categorical import PtdCategorical
from .modules.pytorch_wrapper.ptd_chi2 import PtdChi2
from .modules.pytorch_wrapper.ptd_exponential import PtdExponential
from .modules.pytorch_wrapper.ptd_gamma import PtdGamma
from .modules.pytorch_wrapper.ptd_normal import PtdNormal
from .modules.pytorch_wrapper.ptd_poisson import PtdPoisson
from .modules.pytorch_wrapper.ptd_student_t import PtdStudentT
from .modules.pytorch_wrapper.ptd_uniform import PtdUniform
from .modules.pytorch_wrapper.ptdm_cdf import PtdmCdf
from .modules.pytorch_wrapper.ptdm_cdf_tensor import PtdmCdfTensor
from .modules.pytorch_wrapper.ptdm_icdf import PtdmIcdf
from .modules.pytorch_wrapper.ptdm_icdf_tensor import PtdmIcdfTensor
from .modules.pytorch_wrapper.ptdm_log_prob import PtdmLogProb
from .modules.pytorch_wrapper.ptdm_log_prob_tensor import PtdmLogProbTensor
from .modules.pytorch_wrapper.ptdm_pdf import PtdmPdf
from .modules.pytorch_wrapper.ptdm_pdf_tensor import PtdmPdfTensor
from .modules.pytorch_wrapper.ptdm_pmf import PtdmPmf
from .modules.pytorch_wrapper.ptdm_pmf_tensor import PtdmPmfTensor
from .modules.pytorch_wrapper.ptdm_sample import PtdmSample
from .modules.pytorch_wrapper.ptf_gelu import PtfGELU
from .modules.pytorch_wrapper.ptf_leaky_relu import PtfLeakyReLU
from .modules.pytorch_wrapper.ptf_log_softmax import PtfLogSoftmax
from .modules.pytorch_wrapper.ptf_relu import PtfReLU
from .modules.pytorch_wrapper.ptf_sigmoid import PtfSigmoid
from .modules.pytorch_wrapper.ptf_silu import PtfSiLU
from .modules.pytorch_wrapper.ptf_softmax import PtfSoftmax
from .modules.pytorch_wrapper.ptf_softplus import PtfSoftplus
from .modules.pytorch_wrapper.ptf_tanh import PtfTanh
from .modules.pytorch_wrapper.ptn_avg_pool2d import PtnAvgPool2d
from .modules.pytorch_wrapper.ptn_batch_norm2d import PtnBatchNorm2d
from .modules.pytorch_wrapper.ptn_bce_loss import PtnBCELoss
from .modules.pytorch_wrapper.ptn_bce_with_logits_loss import PtnBCEWithLogitsLoss
from .modules.pytorch_wrapper.ptn_chained_model import PtnChainedModel
from .modules.pytorch_wrapper.ptn_chained_model_with_attention_mask import PtnChainedModelWithAttentionMask
from .modules.pytorch_wrapper.ptn_conv2d import PtnConv2d
from .modules.pytorch_wrapper.ptn_conv_model import PtnConvModel
from .modules.pytorch_wrapper.ptn_cross_entropy_loss import PtnCrossEntropyLoss
from .modules.pytorch_wrapper.ptn_embedding import PtnEmbedding
from .modules.pytorch_wrapper.ptn_embedding_rnn_linear import PtnEmbeddingRNNLinear
from .modules.pytorch_wrapper.ptn_embedding_transformer_linear import PtnEmbeddingTransformerLinear
from .modules.pytorch_wrapper.ptn_gru import PtnGRU
from .modules.pytorch_wrapper.ptn_gru_linear import PtnGRULinear
from .modules.pytorch_wrapper.ptn_hf_fine_tuned_classification_model import PtnHfFineTunedClassificationModel
from .modules.pytorch_wrapper.ptn_hf_lora_classification_model import PtnHfLoraClassificationModel
from .modules.pytorch_wrapper.ptn_huber_loss import PtnHuberLoss
from .modules.pytorch_wrapper.ptn_instance_norm2d import PtnInstanceNorm2d
from .modules.pytorch_wrapper.ptn_kl_div_loss import PtnKLDivLoss
from .modules.pytorch_wrapper.ptn_l1_loss import PtnL1Loss
from .modules.pytorch_wrapper.ptn_layer_norm import PtnLayerNorm
from .modules.pytorch_wrapper.ptn_linear import PtnLinear
from .modules.pytorch_wrapper.ptn_linear_model import PtnLinearModel
from .modules.pytorch_wrapper.ptn_lstm import PtnLSTM
from .modules.pytorch_wrapper.ptn_lstm_linear import PtnLSTMLinear
from .modules.pytorch_wrapper.ptn_masked_mean_pooling import PtnMaskedMeanPooling
from .modules.pytorch_wrapper.ptn_max_pool2d import PtnMaxPool2d
from .modules.pytorch_wrapper.ptn_model_with_closure import PtnModelWithClosure
from .modules.pytorch_wrapper.ptn_mse_loss import PtnMSELoss
from .modules.pytorch_wrapper.ptn_multihead_attention import PtnMultiheadAttention
from .modules.pytorch_wrapper.ptn_multihead_attention_custom import PtnMultiheadAttentionCustom
from .modules.pytorch_wrapper.ptn_nll_loss import PtnNLLLoss
from .modules.pytorch_wrapper.ptn_pre_add_channel_axis import PtnPreAddChannelAxis
from .modules.pytorch_wrapper.ptn_pre_flatten import PtnPreFlatten
from .modules.pytorch_wrapper.ptn_residual_connection_model import PtnResidualConnectionModel
from .modules.pytorch_wrapper.ptn_residual_connection_model_with_attention_mask import PtnResidualConnectionModelWithAttentionMask
from .modules.pytorch_wrapper.ptn_resnet_model import PtnResnetModel
from .modules.pytorch_wrapper.ptn_rnn import PtnRNN
from .modules.pytorch_wrapper.ptn_rnn_linear import PtnRNNLinear
from .modules.pytorch_wrapper.ptn_smooth_l1_loss import PtnSmoothL1Loss
from .modules.pytorch_wrapper.pto_adam import PtoAdam
from .modules.pytorch_wrapper.pto_adamw import PtoAdamW
from .modules.pytorch_wrapper.pto_lr_scheduler_cosine_annealing import PtoLrSchedulerCosineAnnealing
from .modules.pytorch_wrapper.pto_lr_scheduler_reduce_on_plateau import PtoLrSchedulerReduceOnPlateau
from .modules.pytorch_wrapper.pto_lr_scheduler_step import PtoLrSchedulerStep
from .modules.pytorch_wrapper.pto_sgd import PtoSGD
from .modules.pytorch_wrapper.pto_simple import PtoSimple
from .modules.pytorch_wrapper.ptv_dataset import PtvDataset
from .modules.pytorch_wrapper.ptv_dataset_len import PtvDatasetLen
from .modules.pytorch_wrapper.ptv_dataset_loader import PtvDatasetLoader
from .modules.pytorch_wrapper.ptv_hf_dataset_with_token_encode import PtvHfDatasetWithTokenEncode
from .modules.pytorch_wrapper.ptv_hf_glove_dataset import PtvHfGloveDataset
from .modules.pytorch_wrapper.ptv_hf_local_dataset import PtvHfLocalDataset
from .modules.pytorch_wrapper.ptv_image_folder_dataset import PtvImageFolderDataset
from .modules.pytorch_wrapper.ptv_sequential_tensor_dataset import PtvSequentialTensorDataset
from .modules.pytorch_wrapper.ptv_transforms_data_augment import PtvTransformsDataAugment
from .modules.pytorch_wrapper.ptv_transforms_resize import PtvTransformsResize
from .modules.pytorch_wrapper.ptv_transforms_to_tensor import PtvTransformsToTensor
from .modules.sentencepiece_wrapper.sp_encode import SpEncode
from .modules.sentencepiece_wrapper.sp_load_model import SpLoadModel
T = TypeVar("T")


"""
NODE_CLASS_MAPPINGS (Dict[str, Type[T]]):
    A dictionary mapping node names to their corresponding class implementations.
"""

NODE_CLASS_MAPPINGS: Dict[str, Type[T]] = {
    "HfTokenizerEncode": HfTokenizerEncode,
    "PtAbs": PtAbs,
    "PtAcos": PtAcos,
    "PtAdd": PtAdd,
    "PtApplyFunction": PtApplyFunction,
    "PtArange": PtArange,
    "PtArgmax": PtArgmax,
    "PtArgmin": PtArgmin,
    "PtAsin": PtAsin,
    "PtAtan": PtAtan,
    "PtBitwiseAnd": PtBitwiseAnd,
    "PtBitwiseLeftShift": PtBitwiseLeftShift,
    "PtBitwiseNot": PtBitwiseNot,
    "PtBitwiseOr": PtBitwiseOr,
    "PtBitwiseRightShift": PtBitwiseRightShift,
    "PtBitwiseXor": PtBitwiseXor,
    "PtBmm": PtBmm,
    "PtBoolCreate": PtBoolCreate,
    "PtComputeLoss": PtComputeLoss,
    "PtConcat": PtConcat,
    "PtCos": PtCos,
    "PtCosh": PtCosh,
    "PtCrop": PtCrop,
    "PtDataLoader": PtDataLoader,
    "PtDataLoaderFromTensors": PtDataLoaderFromTensors,
    "PtDiv": PtDiv,
    "PtEinsum": PtEinsum,
    "PtEq": PtEq,
    "PtEvaluateClassificationModel": PtEvaluateClassificationModel,
    "PtExp": PtExp,
    "PtFlatten": PtFlatten,
    "PtFloatCreate": PtFloatCreate,
    "PtFloorDiv": PtFloorDiv,
    "PtFromImage": PtFromImage,
    "PtFromImageTranspose": PtFromImageTranspose,
    "PtFromLatent": PtFromLatent,
    "PtFromNumpy": PtFromNumpy,
    "PtFull": PtFull,
    "PtGather": PtGather,
    "PtGe": PtGe,
    "PtGt": PtGt,
    "PtIndexSelect": PtIndexSelect,
    "PtIntCreate": PtIntCreate,
    "PtInterpolateByScaleFactor": PtInterpolateByScaleFactor,
    "PtInterpolateToSize": PtInterpolateToSize,
    "PtLe": PtLe,
    "PtLinspace": PtLinspace,
    "PtLoadModel": PtLoadModel,
    "PtLog": PtLog,
    "PtLogicalAnd": PtLogicalAnd,
    "PtLogicalNot": PtLogicalNot,
    "PtLogicalOr": PtLogicalOr,
    "PtLogicalXor": PtLogicalXor,
    "PtLt": PtLt,
    "PtMaskedSelect": PtMaskedSelect,
    "PtMatMul": PtMatMul,
    "PtMax": PtMax,
    "PtMean": PtMean,
    "PtMedian": PtMedian,
    "PtMin": PtMin,
    "PtMm": PtMm,
    "PtMul": PtMul,
    "PtNe": PtNe,
    "PtNeg": PtNeg,
    "PtOnes": PtOnes,
    "PtPad": PtPad,
    "PtPermute": PtPermute,
    "PtPow": PtPow,
    "PtPredictClassificationModel": PtPredictClassificationModel,
    "PtPredictRegressionModel": PtPredictRegressionModel,
    "PtProd": PtProd,
    "PtRand": PtRand,
    "PtRandInt": PtRandInt,
    "PtRandn": PtRandn,
    "PtRemainder": PtRemainder,
    "PtReshape": PtReshape,
    "PtSaveModel": PtSaveModel,
    "PtScatter": PtScatter,
    "PtShowSize": PtShowSize,
    "PtShowText": PtShowText,
    "PtSin": PtSin,
    "PtSinh": PtSinh,
    "PtSize": PtSize,
    "PtSizeCreate": PtSizeCreate,
    "PtSizeToNumpy": PtSizeToNumpy,
    "PtSizeToString": PtSizeToString,
    "PtSqrt": PtSqrt,
    "PtSqueeze": PtSqueeze,
    "PtStack": PtStack,
    "PtStd": PtStd,
    "PtSub": PtSub,
    "PtSum": PtSum,
    "PtTan": PtTan,
    "PtTanh": PtTanh,
    "PtToBfloat16": PtToBfloat16,
    "PtToFloat16": PtToFloat16,
    "PtToFloat32": PtToFloat32,
    "PtToFloat64": PtToFloat64,
    "PtToImage": PtToImage,
    "PtToImageTranspose": PtToImageTranspose,
    "PtToInt16": PtToInt16,
    "PtToInt32": PtToInt32,
    "PtToInt64": PtToInt64,
    "PtToInt8": PtToInt8,
    "PtToLatent": PtToLatent,
    "PtToNumpy": PtToNumpy,
    "PtToRgbTensors": PtToRgbTensors,
    "PtToUint8": PtToUint8,
    "PtTokenizer": PtTokenizer,
    "PtTrainClassificationModel": PtTrainClassificationModel,
    "PtTrainClassificationModelLr": PtTrainClassificationModelLr,
    "PtTrainClassificationTransformerModel": PtTrainClassificationTransformerModel,
    "PtTrainFineTuneClassificationTransformerModel": PtTrainFineTuneClassificationTransformerModel,
    "PtTrainModel": PtTrainModel,
    "PtTrainRNNModel": PtTrainRNNModel,
    "PtTrainRegressionModel": PtTrainRegressionModel,
    "PtUnsqueeze": PtUnsqueeze,
    "PtVar": PtVar,
    "PtView": PtView,
    "PtWhere": PtWhere,
    "PtZeros": PtZeros,
    "PtdBernoulli": PtdBernoulli,
    "PtdBeta": PtdBeta,
    "PtdBinomial": PtdBinomial,
    "PtdCategorical": PtdCategorical,
    "PtdChi2": PtdChi2,
    "PtdExponential": PtdExponential,
    "PtdGamma": PtdGamma,
    "PtdNormal": PtdNormal,
    "PtdPoisson": PtdPoisson,
    "PtdStudentT": PtdStudentT,
    "PtdUniform": PtdUniform,
    "PtdmCdf": PtdmCdf,
    "PtdmCdfTensor": PtdmCdfTensor,
    "PtdmIcdf": PtdmIcdf,
    "PtdmIcdfTensor": PtdmIcdfTensor,
    "PtdmLogProb": PtdmLogProb,
    "PtdmLogProbTensor": PtdmLogProbTensor,
    "PtdmPdf": PtdmPdf,
    "PtdmPdfTensor": PtdmPdfTensor,
    "PtdmPmf": PtdmPmf,
    "PtdmPmfTensor": PtdmPmfTensor,
    "PtdmSample": PtdmSample,
    "PtfGELU": PtfGELU,
    "PtfLeakyReLU": PtfLeakyReLU,
    "PtfLogSoftmax": PtfLogSoftmax,
    "PtfReLU": PtfReLU,
    "PtfSiLU": PtfSiLU,
    "PtfSigmoid": PtfSigmoid,
    "PtfSoftmax": PtfSoftmax,
    "PtfSoftplus": PtfSoftplus,
    "PtfTanh": PtfTanh,
    "PtnAvgPool2d": PtnAvgPool2d,
    "PtnBCELoss": PtnBCELoss,
    "PtnBCEWithLogitsLoss": PtnBCEWithLogitsLoss,
    "PtnBatchNorm2d": PtnBatchNorm2d,
    "PtnChainedModel": PtnChainedModel,
    "PtnChainedModelWithAttentionMask": PtnChainedModelWithAttentionMask,
    "PtnConv2d": PtnConv2d,
    "PtnConvModel": PtnConvModel,
    "PtnCrossEntropyLoss": PtnCrossEntropyLoss,
    "PtnEmbedding": PtnEmbedding,
    "PtnEmbeddingRNNLinear": PtnEmbeddingRNNLinear,
    "PtnEmbeddingTransformerLinear": PtnEmbeddingTransformerLinear,
    "PtnGRU": PtnGRU,
    "PtnGRULinear": PtnGRULinear,
    "PtnHfFineTunedClassificationModel": PtnHfFineTunedClassificationModel,
    "PtnHfLoraClassificationModel": PtnHfLoraClassificationModel,
    "PtnHuberLoss": PtnHuberLoss,
    "PtnInstanceNorm2d": PtnInstanceNorm2d,
    "PtnKLDivLoss": PtnKLDivLoss,
    "PtnL1Loss": PtnL1Loss,
    "PtnLSTM": PtnLSTM,
    "PtnLSTMLinear": PtnLSTMLinear,
    "PtnLayerNorm": PtnLayerNorm,
    "PtnLinear": PtnLinear,
    "PtnLinearModel": PtnLinearModel,
    "PtnMSELoss": PtnMSELoss,
    "PtnMaskedMeanPooling": PtnMaskedMeanPooling,
    "PtnMaxPool2d": PtnMaxPool2d,
    "PtnModelWithClosure": PtnModelWithClosure,
    "PtnMultiheadAttention": PtnMultiheadAttention,
    "PtnMultiheadAttentionCustom": PtnMultiheadAttentionCustom,
    "PtnNLLLoss": PtnNLLLoss,
    "PtnPreAddChannelAxis": PtnPreAddChannelAxis,
    "PtnPreFlatten": PtnPreFlatten,
    "PtnRNN": PtnRNN,
    "PtnRNNLinear": PtnRNNLinear,
    "PtnResidualConnectionModel": PtnResidualConnectionModel,
    "PtnResidualConnectionModelWithAttentionMask": PtnResidualConnectionModelWithAttentionMask,
    "PtnResnetModel": PtnResnetModel,
    "PtnSmoothL1Loss": PtnSmoothL1Loss,
    "PtoAdam": PtoAdam,
    "PtoAdamW": PtoAdamW,
    "PtoLrSchedulerCosineAnnealing": PtoLrSchedulerCosineAnnealing,
    "PtoLrSchedulerReduceOnPlateau": PtoLrSchedulerReduceOnPlateau,
    "PtoLrSchedulerStep": PtoLrSchedulerStep,
    "PtoSGD": PtoSGD,
    "PtoSimple": PtoSimple,
    "PtvDataset": PtvDataset,
    "PtvDatasetLen": PtvDatasetLen,
    "PtvDatasetLoader": PtvDatasetLoader,
    "PtvHfDatasetWithTokenEncode": PtvHfDatasetWithTokenEncode,
    "PtvHfGloveDataset": PtvHfGloveDataset,
    "PtvHfLocalDataset": PtvHfLocalDataset,
    "PtvImageFolderDataset": PtvImageFolderDataset,
    "PtvSequentialTensorDataset": PtvSequentialTensorDataset,
    "PtvTransformsDataAugment": PtvTransformsDataAugment,
    "PtvTransformsResize": PtvTransformsResize,
    "PtvTransformsToTensor": PtvTransformsToTensor,
    "SpEncode": SpEncode,
    "SpLoadModel": SpLoadModel,
}


"""
NODE_DISPLAY_NAME_MAPPINGS (Dict[str, str]):
    A dictionary mapping node names to user-friendly display names.
"""

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "HfTokenizerEncode": "Hf Tokenizer Encode",
    "PtAbs": "Pt Abs",
    "PtAcos": "Pt Acos",
    "PtAdd": "Pt Add",
    "PtApplyFunction": "Pt Apply Function",
    "PtArange": "Pt Arange",
    "PtArgmax": "Pt Argmax",
    "PtArgmin": "Pt Argmin",
    "PtAsin": "Pt Asin",
    "PtAtan": "Pt Atan",
    "PtBitwiseAnd": "Pt Bitwise And",
    "PtBitwiseLeftShift": "Pt Bitwise Left Shift",
    "PtBitwiseNot": "Pt Bitwise Not",
    "PtBitwiseOr": "Pt Bitwise Or",
    "PtBitwiseRightShift": "Pt Bitwise Right Shift",
    "PtBitwiseXor": "Pt Bitwise Xor",
    "PtBmm": "Pt Bmm",
    "PtBoolCreate": "Pt Bool Create",
    "PtComputeLoss": "Pt Compute Loss",
    "PtConcat": "Pt Concat",
    "PtCos": "Pt Cos",
    "PtCosh": "Pt Cosh",
    "PtCrop": "Pt Crop",
    "PtDataLoader": "Pt Data Loader",
    "PtDataLoaderFromTensors": "Pt Data Loader From Tensors",
    "PtDiv": "Pt Div",
    "PtEinsum": "Pt Einsum",
    "PtEq": "Pt Eq",
    "PtEvaluateClassificationModel": "Pt Evaluate Classification Model",
    "PtExp": "Pt Exp",
    "PtFlatten": "Pt Flatten",
    "PtFloatCreate": "Pt Float Create",
    "PtFloorDiv": "Pt Floor Div",
    "PtFromImage": "Pt From Image",
    "PtFromImageTranspose": "Pt From Image Transpose",
    "PtFromLatent": "Pt From Latent",
    "PtFromNumpy": "Pt From Numpy",
    "PtFull": "Pt Full",
    "PtGather": "Pt Gather",
    "PtGe": "Pt Ge",
    "PtGt": "Pt Gt",
    "PtIndexSelect": "Pt Index Select",
    "PtIntCreate": "Pt Int Create",
    "PtInterpolateByScaleFactor": "Pt Interpolate By Scale Factor",
    "PtInterpolateToSize": "Pt Interpolate To Size",
    "PtLe": "Pt Le",
    "PtLinspace": "Pt Linspace",
    "PtLoadModel": "Pt Load Model",
    "PtLog": "Pt Log",
    "PtLogicalAnd": "Pt Logical And",
    "PtLogicalNot": "Pt Logical Not",
    "PtLogicalOr": "Pt Logical Or",
    "PtLogicalXor": "Pt Logical Xor",
    "PtLt": "Pt Lt",
    "PtMaskedSelect": "Pt Masked Select",
    "PtMatMul": "Pt Mat Mul",
    "PtMax": "Pt Max",
    "PtMean": "Pt Mean",
    "PtMedian": "Pt Median",
    "PtMin": "Pt Min",
    "PtMm": "Pt Mm",
    "PtMul": "Pt Mul",
    "PtNe": "Pt Ne",
    "PtNeg": "Pt Neg",
    "PtOnes": "Pt Ones",
    "PtPad": "Pt Pad",
    "PtPermute": "Pt Permute",
    "PtPow": "Pt Pow",
    "PtPredictClassificationModel": "Pt Predict Classification Model",
    "PtPredictRegressionModel": "Pt Predict Regression Model",
    "PtProd": "Pt Prod",
    "PtRand": "Pt Rand",
    "PtRandInt": "Pt Rand Int",
    "PtRandn": "Pt Randn",
    "PtRemainder": "Pt Remainder",
    "PtReshape": "Pt Reshape",
    "PtSaveModel": "Pt Save Model",
    "PtScatter": "Pt Scatter",
    "PtShowSize": "Pt Show Size",
    "PtShowText": "Pt Show Text",
    "PtSin": "Pt Sin",
    "PtSinh": "Pt Sinh",
    "PtSize": "Pt Size",
    "PtSizeCreate": "Pt Size Create",
    "PtSizeToNumpy": "Pt Size To Numpy",
    "PtSizeToString": "Pt Size To String",
    "PtSqrt": "Pt Sqrt",
    "PtSqueeze": "Pt Squeeze",
    "PtStack": "Pt Stack",
    "PtStd": "Pt Std",
    "PtSub": "Pt Sub",
    "PtSum": "Pt Sum",
    "PtTan": "Pt Tan",
    "PtTanh": "Pt Tanh",
    "PtToBfloat16": "Pt To Bfloat16",
    "PtToFloat16": "Pt To Float16",
    "PtToFloat32": "Pt To Float32",
    "PtToFloat64": "Pt To Float64",
    "PtToImage": "Pt To Image",
    "PtToImageTranspose": "Pt To Image Transpose",
    "PtToInt16": "Pt To Int16",
    "PtToInt32": "Pt To Int32",
    "PtToInt64": "Pt To Int64",
    "PtToInt8": "Pt To Int8",
    "PtToLatent": "Pt To Latent",
    "PtToNumpy": "Pt To Numpy",
    "PtToRgbTensors": "Pt To Rgb Tensors",
    "PtToUint8": "Pt To Uint8",
    "PtTokenizer": "Pt Tokenizer",
    "PtTrainClassificationModel": "Pt Train Classification Model",
    "PtTrainClassificationModelLr": "Pt Train Classification Model Lr",
    "PtTrainClassificationTransformerModel": "Pt Train Classification Transformer Model",
    "PtTrainFineTuneClassificationTransformerModel": "Pt Train Fine Tune Classification Transformer Model",
    "PtTrainModel": "Pt Train Model",
    "PtTrainRNNModel": "Pt Train RNN Model",
    "PtTrainRegressionModel": "Pt Train Regression Model",
    "PtUnsqueeze": "Pt Unsqueeze",
    "PtVar": "Pt Var",
    "PtView": "Pt View",
    "PtWhere": "Pt Where",
    "PtZeros": "Pt Zeros",
    "PtdBernoulli": "Ptd Bernoulli",
    "PtdBeta": "Ptd Beta",
    "PtdBinomial": "Ptd Binomial",
    "PtdCategorical": "Ptd Categorical",
    "PtdChi2": "Ptd Chi2",
    "PtdExponential": "Ptd Exponential",
    "PtdGamma": "Ptd Gamma",
    "PtdNormal": "Ptd Normal",
    "PtdPoisson": "Ptd Poisson",
    "PtdStudentT": "Ptd Student T",
    "PtdUniform": "Ptd Uniform",
    "PtdmCdf": "Ptdm Cdf",
    "PtdmCdfTensor": "Ptdm Cdf Tensor",
    "PtdmIcdf": "Ptdm Icdf",
    "PtdmIcdfTensor": "Ptdm Icdf Tensor",
    "PtdmLogProb": "Ptdm Log Prob",
    "PtdmLogProbTensor": "Ptdm Log Prob Tensor",
    "PtdmPdf": "Ptdm Pdf",
    "PtdmPdfTensor": "Ptdm Pdf Tensor",
    "PtdmPmf": "Ptdm Pmf",
    "PtdmPmfTensor": "Ptdm Pmf Tensor",
    "PtdmSample": "Ptdm Sample",
    "PtfGELU": "Ptf GELU",
    "PtfLeakyReLU": "Ptf Leaky ReLU",
    "PtfLogSoftmax": "Ptf Log Softmax",
    "PtfReLU": "Ptf ReLU",
    "PtfSiLU": "Ptf SiLU",
    "PtfSigmoid": "Ptf Sigmoid",
    "PtfSoftmax": "Ptf Softmax",
    "PtfSoftplus": "Ptf Softplus",
    "PtfTanh": "Ptf Tanh",
    "PtnAvgPool2d": "Ptn Avg Pool 2d",
    "PtnBCELoss": "Ptn BCE Loss",
    "PtnBCEWithLogitsLoss": "Ptn BCE With Logits Loss",
    "PtnBatchNorm2d": "Ptn Batch Norm 2d",
    "PtnChainedModel": "Ptn Chained Model",
    "PtnChainedModelWithAttentionMask": "Ptn Chained Model With Attention Mask",
    "PtnConv2d": "Ptn Conv 2d",
    "PtnConvModel": "Ptn Conv Model",
    "PtnCrossEntropyLoss": "Ptn Cross Entropy Loss",
    "PtnEmbedding": "Ptn Embedding",
    "PtnEmbeddingRNNLinear": "Ptn Embedding RNN Linear",
    "PtnEmbeddingTransformerLinear": "Ptn Embedding Transformer Linear",
    "PtnGRU": "Ptn GRU",
    "PtnGRULinear": "Ptn GRU Linear",
    "PtnHfFineTunedClassificationModel": "Ptn Hf Fine Tuned Classification Model",
    "PtnHfLoraClassificationModel": "Ptn Hf Lora Classification Model",
    "PtnHuberLoss": "Ptn Huber Loss",
    "PtnInstanceNorm2d": "Ptn Instance Norm 2d",
    "PtnKLDivLoss": "Ptn KL Div Loss",
    "PtnL1Loss": "Ptn L1 Loss",
    "PtnLSTM": "Ptn LSTM",
    "PtnLSTMLinear": "Ptn LSTM Linear",
    "PtnLayerNorm": "Ptn Layer Norm",
    "PtnLinear": "Ptn Linear",
    "PtnLinearModel": "Ptn Linear Model",
    "PtnMSELoss": "Ptn MSE Loss",
    "PtnMaskedMeanPooling": "Ptn Masked Mean Pooling",
    "PtnMaxPool2d": "Ptn Max Pool 2d",
    "PtnModelWithClosure": "Ptn Model With Closure",
    "PtnMultiheadAttention": "Ptn Multihead Attention",
    "PtnMultiheadAttentionCustom": "Ptn Multihead Attention Custom",
    "PtnNLLLoss": "Ptn NLL Loss",
    "PtnPreAddChannelAxis": "Ptn Pre Add Channel Axis",
    "PtnPreFlatten": "Ptn Pre Flatten",
    "PtnRNN": "Ptn RNN",
    "PtnRNNLinear": "Ptn RNN Linear",
    "PtnResidualConnectionModel": "Ptn Residual Connection Model",
    "PtnResidualConnectionModelWithAttentionMask": "Ptn Residual Connection Model With Attention Mask",
    "PtnResnetModel": "Ptn Resnet Model",
    "PtnSmoothL1Loss": "Ptn Smooth L1 Loss",
    "PtoAdam": "Pto Adam",
    "PtoAdamW": "Pto AdamW",
    "PtoLrSchedulerCosineAnnealing": "Pto Lr Scheduler Cosine Annealing",
    "PtoLrSchedulerReduceOnPlateau": "Pto Lr Scheduler Reduce On Plateau",
    "PtoLrSchedulerStep": "Pto Lr Scheduler Step",
    "PtoSGD": "Pto SGD",
    "PtoSimple": "Pto Simple",
    "PtvDataset": "Ptv Dataset",
    "PtvDatasetLen": "Ptv Dataset Len",
    "PtvDatasetLoader": "Ptv Dataset Loader",
    "PtvHfDatasetWithTokenEncode": "Ptv Hf Dataset With Token Encode",
    "PtvHfGloveDataset": "Ptv Hf Glove Dataset",
    "PtvHfLocalDataset": "Ptv Hf Local Dataset",
    "PtvImageFolderDataset": "Ptv Image Folder Dataset",
    "PtvSequentialTensorDataset": "Ptv Sequential Tensor Dataset",
    "PtvTransformsDataAugment": "Ptv Transforms Data Augment",
    "PtvTransformsResize": "Ptv Transforms Resize",
    "PtvTransformsToTensor": "Ptv Transforms To Tensor",
    "SpEncode": "Sp Encode",
    "SpLoadModel": "Sp Load Model",
}


"""
Below two lines were taken from:
https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/__init__.py
See credit/credit.md for the full license.
"""

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
