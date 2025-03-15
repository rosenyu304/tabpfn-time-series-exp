from tabpfn_time_series.experimental.pipeline.pipeline import TabPFNTSPipeline
from tabpfn_time_series.experimental.multivariate.ar_multivariate_pipeline import (
    TabPFNARMultiVariatePipeline,
)
from tabpfn_time_series.experimental.multivariate.fusion_multivariate_pipeline import (
    TabPFNFusionMultiVariatePipeline,
)

PIPELINE_MAPPING = {
    "TabPFNTSPipeline": TabPFNTSPipeline,
    "TabPFNARMultiVariatePipeline": TabPFNARMultiVariatePipeline,
    "TabPFNFusionMultiVariatePipeline": TabPFNFusionMultiVariatePipeline,
}
