from tabpfn_time_series.experimental.pipeline.pipeline import TabPFNTSPipeline
from tabpfn_time_series.experimental.multivariate.ar_multivariate_pipeline import (
    TabPFNARMultiVariatePipeline,
)

PIPELINE_MAPPING = {
    "TabPFNTSPipeline": TabPFNTSPipeline,
    "TabPFNARMultiVariatePipeline": TabPFNARMultiVariatePipeline,
}
