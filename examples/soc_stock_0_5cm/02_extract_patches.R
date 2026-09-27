# ══════════════════════════════════════════════════════════════════════════════
# 02 -- folded into 01 on 2026-09-26
#
# Stage 02 cut the patches out of the rasters for the points stage 01 had
# prepared. Both are now one call, dsm_prepare() (R/prepare.R), made from
# 01_prepare_dataset.R -- and _p1_prepare_check.R proved on this data that it
# builds the identical store: the three windows bit for bit, every table value
# for value, 19 of 19 checks.
#
# This file is kept so that anyone following the old order -- 01, then 02 --
# is told where the work went instead of meeting "file not found". It runs
# nothing.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/02_extract_patches.R")
# ══════════════════════════════════════════════════════════════════════════════

stop("Stage 02 is now part of stage 01.\n",
     "  01_prepare_dataset.R builds the point table AND the patch store, with ",
     "dsm_prepare().\n  If 01 has run, the store is ready: go on to ",
     "99_check_pipeline.R, then 03_run_tuning.R.", call. = FALSE)
