#!/usr/bin/env bash
# Interpretability re-runs for the revision, on the corrected-R-peak data.
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2
GQ_DATA=data_fusion_fix GQ_MODEL=results/external_fix/model_paf.keras \
  GQ_OUT=results/gradcam_quant_fix ./run_gpu.sh gradcam_quant.py
GQ_DATA=data_fusion_fix GQ_MODEL=results/external_fix/model_paf.keras \
  ./run_gpu.sh xai_faithfulness.py
echo XAI_DONE
