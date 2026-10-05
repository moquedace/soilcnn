# The numbers the vignette's text reads from a run of the SOC 0-30 cm trial:
# every [DRAFT] mark of vignettes/soilcnn.Rmd, in the order of the text, with
# the value it asks for. The text is then filled from the run in one pass, and
# no number is copied into it by hand from a table.
#
#   results_from <- "smoke"           # the run: "sample10" when it is done
#   maps_from <- "maps/check_tile"    # optional: 05a's tiles stand in for 05b's maps
#   source("D:/usuario_armazenamento/cassio/projects/soilcnn/tools/vignette_numbers.R")
#
# Reads the run's files: 03's tuning comparison, 04's intervals, 06's
# comparison, a regular sample of the spatial design's map and 07's importance
# -- what the figures are drawn from (tools/vignette_results.R), with Figure
# 9's own block bootstrap. What a run does not have yet is said, never
# guessed, and a part that fails says why and leaves the others to run. The
# printout is kept in <run>/vignette/vignette_numbers.txt.

if (!exists("results_from")) results_from <- "smoke"
if (!exists("maps_from")) maps_from <- "maps"
trial <- "D:/usuario_armazenamento/cassio/projects/soc_stock_0_30cm_lac"
run   <- file.path(trial, "outputs", results_from)
stopifnot(dir.exists(run))
suppressMessages(library(soilcnn))
csv <- function(p) utils::read.csv2(p, stringsAsFactors = FALSE)
# 00_settings.R's order -- and so 07's, whose comparison columns i1..i5 follow it.
designs <- c("spatial", "knndm", "random", "holdout", "region")
kept <- file.path(run, "vignette")
dir.create(kept, showWarnings = FALSE)

printed <- character(0)
say <- function(...) {
  s <- paste0(...)
  printed <<- c(printed, s)
  cat(s, "\n", sep = "")
}
mark <- function(what, value) say(sprintf("  %-40s %s", paste0("[", what, "]"), value))
num <- function(x, d = 3) ifelse(is.finite(x), formatC(x, digits = d, format = "f", big.mark = ","), "NA")
# A CCC, its SE, a gap between two: four decimals. Not ccc(), soilcnn's metric.
ccc_text <- function(x) num(x, 4)
pct <- function(x, d = 1) ifelse(is.finite(x), paste0(formatC(100 * x, digits = d, format = "f"), "%"), "NA")
span <- function(x, f = num) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return("NA")
  if (min(x) == max(x)) f(min(x)) else paste(f(min(x)), "to", f(max(x)))
}
# A part that fails says so, and the next one runs.
part <- function(title, expr) {
  say("\n== ", title, " ", strrep("=", max(3L, 74L - nchar(title))))
  tryCatch(expr, error = function(e) say("  NOT READ: ", conditionMessage(e)))
  invisible(NULL)
}
final_cfg <- function(d) {
  readRDS(file.path(run, "final_model", d, "comparison", "final_run_summary.rds"))$selected_config_ids[1]
}

say("Vignette numbers from '", results_from, "' (", run, "), ", format(Sys.time(), "%Y-%m-%d %H:%M"),
    ", soilcnn ", as.character(utils::packageVersion("soilcnn")))
if (identical(results_from, "smoke")) {
  say("THE SMOKE'S MODELS NEVER LEARNED: these numbers test the script, not the text.")
}

# ── Figure 8: the selection (section 4) ───────────────────────────────────────
part("Figure 8 -- the configurations the spatial design tried (section 4)", {
  tun  <- file.path(run, "tuning", "spatial")
  bc   <- csv(file.path(tun, "comparison", "comparison_by_config.csv"))
  grid <- csv(file.path(tun, "tune_grid.csv"))
  sel  <- readRDS(file.path(tun, "comparison", "selection.rds"))$config_id[1]
  best   <- which.max(bc$val_ccc_mean)
  chosen <- match(sel, bc$config_id)
  threshold <- bc$val_ccc_mean[best] - bc$val_ccc_se[best]
  gap <- bc$val_ccc_mean[best] - bc$val_ccc_mean[chosen]
  windows_of <- function(id) gsub("_", " + ", as.character(grid$window_sizes[match(id, grid$config_id)]))
  nf <- seed_noise_floor(csv(file.path(tun, "comparison", "comparison_all.csv")))
  say(sprintf("  %d configuration(s) -- the text says thirty -- over %d fold(s) x %d seed(s)",
              nrow(bc), max(bc$n_folds), max(bc$n_seeds)))
  if (best == chosen) say("  NOTE: the best mean IS the choice, and the paragraph says it is not")
  mark("config and CCC", sprintf("%s, CCC %s (SE %s), windows %s", bc$config_id[best],
                                 ccc_text(bc$val_ccc_mean[best]), ccc_text(bc$val_ccc_se[best]),
                                 windows_of(bc$config_id[best])))
  mark("value (the threshold)", ccc_text(threshold))
  mark("n (the configurations reaching it)", sum(bc$val_ccc_mean >= threshold, na.rm = TRUE))
  mark("config, windows and parameters", sprintf("%s, windows %s, %s M parameters", sel,
                                                 windows_of(sel), num(bc$n_params[chosen] / 1e6, 2)))
  mark("k (times fewer parameters)", num(bc$n_params[best] / bc$n_params[chosen], 1))
  mark("difference (CCC)", ccc_text(gap))
  mark("value (the noise floor)", sprintf("%s, the median sd between seeds (widest range %s; %d config x fold)",
                                          ccc_text(nf$median_sd), ccc_text(nf$max_range), nf$n_comparable))
  if (is.finite(nf$median_sd)) {
    say("  check: the difference is ", if (gap <= nf$median_sd) "within" else "ABOVE",
        " the noise floor (the text: 'a difference the seeds alone produce')")
  }
})

# ── Figure 9: the five designs on the one test set (section 6) ────────────────
part("Figure 9 -- five designs, one test set (section 6)", {
  models <- csv(file.path(run, "comparison", "models.csv"))
  mark("range (every design's MAE)", paste(span(models$test_mae, function(x) num(x, 1)), "t/ha"))
  for (i in seq_len(nrow(models))) {
    say(sprintf("    %-8s MAE %s t/ha, CCC %s; cross-validated CCC %s, optimism %s", models$design[i],
                num(models$test_mae[i], 1), ccc_text(models$test_ccc[i]), ccc_text(models$cv_ccc[i]),
                ccc_text(models$optimism_ccc[i])))
  }
  # Figure 9's own intervals: whole 100 km blocks, every profile one vote, its seed.
  meta <- csv(file.path(run, "patches", "patch_meta.csv"))
  test <- lapply(designs, function(d) {
    e <- csv(file.path(run, "final_model", d, final_cfg(d), "ensemble_predictions.csv"))
    e <- e[e$dataset_role == "test", c("sample_id", "obs", "pred")]
    e[order(e$sample_id), ]
  })
  names(test) <- designs
  stopifnot(all(vapply(test, function(e) identical(e$sample_id, test$spatial$sample_id), logical(1))))
  say("  ", nrow(test$spatial), " test profiles, the same for the five")
  at  <- match(test$spatial$sample_id, meta$sample_id)
  blk <- equal_area_blocks(meta$x[at], meta$y[at], size_km = 100)
  paired <- do.call(rbind, lapply(setdiff(designs, "spatial"), function(d) {
    b <- block_bootstrap(test$spatial$obs, test[[d]]$pred, blk, against = test$spatial$pred,
                         metric = "mae", weights = "profile", n_boot = 2000L, seed = 20261004L)
    data.frame(design = d, diff = b$difference, lo = b$ci_low, hi = b$ci_high)
  }))
  excl <- paired$design[paired$lo > 0 | paired$hi < 0]
  mark("range (the differences against spatial)", paste(span(paired$diff, function(x) num(x, 2)), "t/ha"))
  mark("which (intervals that exclude zero)", if (length(excl)) paste(excl, collapse = ", ") else "none")
  for (i in seq_len(nrow(paired))) {
    say(sprintf("    %-8s %s t/ha, 95%% from whole blocks %s to %s%s", paired$design[i],
                num(paired$diff[i], 2), num(paired$lo[i], 2), num(paired$hi[i], 2),
                if (paired$design[i] %in% excl) "  (excludes 0)" else ""))
  }
  row_of <- function(d) models[models$design == d, , drop = FALSE]
  mark("value (random: cross-validated CCC)", ccc_text(row_of("random")$cv_ccc))
  mark("value (random: test CCC)", ccc_text(row_of("random")$test_ccc))
  mark("value (spatial: cross-validated CCC)", ccc_text(row_of("spatial")$cv_ccc))
  mark("value (spatial: test CCC)", ccc_text(row_of("spatial")$test_ccc))
})

# ── Figure 10: the intervals on the one test set (section 7) ──────────────────
part("Figure 10 -- the 90% intervals on the test set (section 7)", {
  iv <- csv(file.path(run, "final_model", "intervals_summary.csv"))
  pt <- iv[iv$level == "pi90" & iv$weighting == "point", ]
  say("  closest to 90%, by point:")
  for (d in designs) {
    r <- pt[pt$design == d, , drop = FALSE]
    if (nrow(r) == 0L) next
    r <- r[order(abs(r$coverage - 0.9), -r$coverage), , drop = FALSE]
    say(sprintf("    %-8s %s %s: %s, mean width %s t/ha", d, r$method[1], r$width[1],
                pct(r$coverage[1]), num(r$mean_width[1], 1)))
  }
  mark("which came closest, by design", "the lines above")
  for (m in c("split", "cv", "cv_plus")) {
    mark(sprintf("range (%s, both widths)", m), span(pt$coverage[pt$method == m], pct))
  }
  w <- merge(pt[pt$width == "constant", c("design", "method", "mean_width", "coverage")],
             pt[pt$width == "level_di", c("design", "method", "mean_width", "coverage")],
             by = c("design", "method"), suffixes = c("_c", "_l"))
  w$change <- w$mean_width_l / w$mean_width_c - 1
  mark("how much (level+DI against constant)",
       paste("mean width", span(w$change, function(x) sprintf("%+.0f%%", 100 * x))))
  for (i in seq_len(nrow(w))) {
    say(sprintf("    %-8s %-8s constant %s t/ha (%s), level+DI %s t/ha (%s): %+.0f%%", w$design[i],
                w$method[i], num(w$mean_width_c[i], 1), pct(w$coverage_c[i]),
                num(w$mean_width_l[i], 1), pct(w$coverage_l[i]), 100 * w$change[i]))
  }
  # Every design's coverage by stratum, as dsm_final() wrote it.
  ct <- do.call(rbind, lapply(designs, function(d) {
    f <- file.path(run, "final_model", d, final_cfg(d), "intervals", "coverage_test.csv")
    if (!file.exists(f)) return(NULL)
    x <- csv(f)
    x$design <- d
    x[x$level == "pi90" & x$weighting == "point", ]
  }))
  fifths <- ct[ct$stratum_type == "level" & ct$width == "level_di", ]
  fifths$k <- as.integer(sub("fifth ", "", fifths$stratum, fixed = TRUE))
  say("  level+DI by fifth of the predicted stock, coverage (profiles):")
  for (d in designs) for (m in c("cv", "split", "cv_plus")) {
    r <- fifths[fifths$design == d & fifths$method == m, , drop = FALSE]
    if (nrow(r) == 0L) next
    r <- r[order(r$k), ]
    say(sprintf("    %-8s %-8s %s", d, m,
                paste(sprintf("%s %s (%d)", r$stratum, pct(r$coverage, 0), r$n), collapse = " | ")))
  }
  lowest  <- fifths[fifths$k == 1L, ]
  highest <- do.call(rbind, lapply(split(fifths, paste(fifths$design, fifths$method)),
                                   function(r) r[which.max(r$k), , drop = FALSE]))
  mark("lowest and highest (level+DI fifths)",
       sprintf("lowest fifth %s; highest %s", span(lowest$coverage, pct), span(highest$coverage, pct)))
  out <- ct[ct$stratum_type == "aoa" & ct$stratum == "outside", ]
  if (nrow(out) == 0L) {
    mark("value (coverage outside the AOA)", "no test profile outside the AOA, in any design")
  } else {
    mark("value (coverage outside the AOA)", span(out$coverage, pct))
    for (d in unique(out$design)) {
      r <- out[out$design == d, ]
      say(sprintf("    %-8s %d profile(s) outside: %s", d, r$n[1],
                  paste(sprintf("%s %s %s", r$method, r$width, pct(r$coverage, 0)), collapse = "; ")))
    }
  }
})

# ── Figure 11: the map (section 8) ────────────────────────────────────────────
#
# A regular sample of the spatial design's map, each cell with its WWF
# ecoregion (Olson et al. 2001) and biome, read as 02_prepare.R reads them:
# s2 off, the rings made valid.
biome_names <- c("1" = "tropical moist broadleaf forests", "2" = "tropical dry broadleaf forests",
                 "3" = "tropical coniferous forests", "4" = "temperate broadleaf and mixed forests",
                 "5" = "temperate conifer forests", "6" = "boreal forests",
                 "7" = "tropical grasslands, savannas and shrublands",
                 "8" = "temperate grasslands, savannas and shrublands",
                 "9" = "flooded grasslands and savannas", "10" = "montane grasslands and shrublands",
                 "11" = "tundra", "12" = "Mediterranean forests, woodlands and scrub",
                 "13" = "deserts and xeric shrublands", "14" = "mangroves")
part("Figure 11 -- the spatial design's map (section 8)", {
  md <- file.path(run, maps_from, "spatial")
  if (!file.exists(file.path(md, "bands.csv"))) stop("no map at ", md, " yet (05b)")
  if (!identical(maps_from, "maps")) say("  REHEARSAL: ", maps_from, " stands in for the region's map")
  bands <- csv(file.path(md, "bands.csv"))
  if (!"width" %in% names(bands)) {          # a map from before 2026-10-05
    bands$width  <- bands$method
    bands$method <- ifelse(bands$kind == "interval", "cv", NA)
  }
  band <- function(kind, method = NA, width = NA, side = NA) {
    hit <- bands$kind == kind & (is.na(method) | bands$method %in% method) &
      (is.na(width) | bands$width %in% width) & (is.na(side) | bands$side %in% side) &
      (kind != "interval" | bands$label %in% "pi90")
    if (sum(hit) != 1L) stop("the map has ", sum(hit), " band(s) of kind '", kind, "' here")
    terra::rast(file.path(md, paste0(bands$band[hit], ".vrt")))
  }
  shown <- if (any(bands$method %in% "split")) "split" else "cv"     # as Figure 11
  med <- terra::rast(file.path(md, "ensemble_median.vrt"))
  xy  <- as.matrix(prediction_sample(med, size = 2e5))
  val <- function(r) { e <- terra::extract(r, xy); e[[ncol(e)]] }
  s <- data.frame(x = xy[, 1], y = xy[, 2], median = val(med),
                  width = val(band("interval", shown, "level_di", "upper")) -
                    val(band("interval", shown, "level_di", "lower")),
                  di = val(band("di")), aoa = val(band("aoa")))
  s <- s[is.finite(s$median), ]
  quant <- function(v, p) stats::quantile(v, p, na.rm = TRUE, names = FALSE)   # not q(): that quits R
  say(sprintf("  %s cells of the map, a regular sample", format(nrow(s), big.mark = ",")))
  mark("range (the median)", sprintf("%s to %s t/ha (2%% and 98%% of the cells); %s to %s at the extremes",
                                     num(quant(s$median, .02), 1), num(quant(s$median, .98), 1),
                                     num(min(s$median), 1), num(max(s$median), 1)))

  s2_was <- suppressMessages(sf::sf_use_s2(FALSE))
  eco <- sf::st_read(file.path(trial, "data", "raw", "ecoregions", "wwf_terr_ecos.shp"), quiet = TRUE)
  eco <- sf::st_make_valid(eco[!eco$ECO_ID %in% c(-9998, -9999), c("ECO_NAME", "BIOME")])
  pts <- sf::st_as_sf(s[, c("x", "y")], coords = c("x", "y"), crs = 4326)
  eco <- sf::st_transform(eco, sf::st_crs(pts))
  hit <- vapply(suppressMessages(sf::st_intersects(pts, eco)), function(i) if (length(i)) i[1] else NA_integer_, integer(1))
  suppressMessages(sf::sf_use_s2(s2_was))
  s$ecoregion <- eco$ECO_NAME[hit]
  s$biome <- unname(biome_names[as.character(eco$BIOME[hit])])
  by_name <- function(g, min_n) {
    ok <- !is.na(g)
    a <- stats::aggregate(data.frame(median = s$median[ok], width = s$width[ok],
                                     outside = 1 - s$aoa[ok]),
                          list(name = g[ok]), mean, na.rm = TRUE)
    a$n <- as.integer(table(g[ok])[a$name])
    a[a$n >= min_n, ]
  }
  eco_tab <- by_name(s$ecoregion, 200L)
  eco_tab <- eco_tab[order(-eco_tab$median), ]
  say("  ", nrow(eco_tab), " ecoregion(s) with 200 sampled cells or more")
  top    <- utils::head(eco_tab, 5L)
  bottom <- eco_tab[rev(utils::tail(seq_len(nrow(eco_tab)), 5L)), ]
  mark("regions (the highest stocks)", paste(sprintf("%s %s", top$name, num(top$median, 0)), collapse = "; "))
  mark("regions (the lowest stocks)", paste(sprintf("%s %s", bottom$name, num(bottom$median, 0)), collapse = "; "))
  bio <- by_name(s$biome, 1L)
  bio <- bio[order(-bio$median), ]
  say("  by biome: median, interval width, share outside the AOA (cells):")
  for (i in seq_len(nrow(bio))) {
    say(sprintf("    %-46s %s t/ha, width %s, outside %s (%s)", bio$name[i], num(bio$median[i], 1),
                num(bio$width[i], 1), pct(bio$outside[i], 0), format(bio$n[i], big.mark = ",")))
  }
  mark("width range", sprintf("%s to %s t/ha (2%% and 98%% of the cells), the %s interval",
                              num(quant(s$width, .02), 1), num(quant(s$width, .98), 1), shown))
  say(sprintf("  check: Spearman of the width with the median %s, with the DI %s (the text: widest where both are high)",
              num(stats::cor(s$width, s$median, method = "spearman", use = "complete.obs"), 2),
              num(stats::cor(s$width, s$di, method = "spearman", use = "complete.obs"), 2)))
  mark("share (inside the AOA)", pct(mean(s$aoa == 1, na.rm = TRUE)))
  far <- eco_tab[order(-eco_tab$outside), ]
  far <- utils::head(far[far$outside > 0, ], 5L)
  mark("where (outside the AOA)", if (nrow(far)) {
    paste(sprintf("%s %s", far$name, pct(far$outside, 0)), collapse = "; ")
  } else "every sampled ecoregion inside")
  iv <- csv(file.path(run, "final_model", "intervals_summary.csv"))
  r <- iv[iv$design == "spatial" & iv$level == "pi90" & iv$method == shown &
            iv$width == "level_di" & iv$weighting == "point", ]
  mark("value (its coverage on the test)", if (nrow(r)) pct(r$coverage[1]) else "NA")
})

# ── Figure 12: what the model learned (section 9) ─────────────────────────────
part("Figure 12 -- what the model learned (section 9)", {
  imp <- file.path(run, "importance")
  for (m in c("shap", "permutation")) {
    f <- file.path(imp, paste0("compare_", m, ".csv"))
    if (!file.exists(f)) {
      say("  ", m, ": no ", basename(f), " yet (07_importance.R)")
      next
    }
    tab <- csv(f)
    rk  <- tab[, grep("^rank_i[0-9]+$", names(tab)), drop = FALSE]
    names(rk) <- designs[as.integer(sub("rank_i", "", names(rk), fixed = TRUE))]
    pairs <- utils::combn(names(rk), 2)
    rho <- apply(pairs, 2, function(p) stats::cor(rk[[p[1]]], rk[[p[2]]], method = "spearman",
                                                   use = "pairwise.complete.obs"))
    worst <- pairs[, which.min(rho)]
    mark(sprintf("%s: Spearman between designs", m),
         sprintf("%s to %s; the lowest %s and %s", num(min(rho, na.rm = TRUE), 2),
                 num(max(rho, na.rm = TRUE), 2), worst[1], worst[2]))
    tab <- tab[order(tab$mean_rank), ]
    mark(sprintf("%s: the ranking (mean rank)", m),
         paste(sprintf("%s %s", tab$target, num(tab$mean_rank, 1)), collapse = "; "))
    shares <- tab[, grep("^share_i[0-9]+$", names(tab)), drop = FALSE]
    none <- tab$target[apply(shares, 1, function(v) all(!is.finite(v) | v < 0.01))]
    say("    under 1% of the leading theme in every design: ",
        if (length(none)) paste(none, collapse = ", ") else "none")
  }
})

writeLines(printed, file.path(kept, "vignette_numbers.txt"), useBytes = TRUE)
message("\nKept in ", file.path(kept, "vignette_numbers.txt"))
