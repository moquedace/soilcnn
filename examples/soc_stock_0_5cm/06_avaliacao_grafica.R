# Graphical evaluation of the CNN -- base R only, nothing to install.
# RStudio: source("examples/soc_stock_0_5cm/06_avaliacao_grafica.R", encoding="UTF-8")
# Console: Rscript gerar_graficos.R "PROJECT_DIR" "OUTPUT_DIR"
# Does not train models and does not modify the source results.

gerar_graficos_cnn <- function(
  project_root = "D:/usuario_armazenamento/cassio/R/deep_learning_caret",
  output_dir = file.path(project_root, "outputs", "avaliacao_grafica"),
  final_run_id = "latest",
  config_id = "auto"
) {
  if (.Platform$OS.type == "windows") {
    local_antigo <- Sys.getlocale("LC_CTYPE")
    suppressWarnings(Sys.setlocale("LC_CTYPE", "Portuguese_Brazil.utf8"))
    on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", local_antigo)), add=TRUE)
  }
  ler <- function(f) read.csv2(f, stringsAsFactors=FALSE, check.names=FALSE)
  target <- "soc_stock_0_5cm"
  # RESOLVED, NOT REMEMBERED.
  #
  # These two used to default to "final_20260823_001856" and "cfg_014", a run
  # and a config that no longer exist -- so the script stopped on its second
  # line for anyone who did not know to pass arguments. A default that names one
  # particular past run is a default that is wrong from the day after it is
  # written. 05 and 07 resolve the same two things the same way.
  final_base <- file.path(project_root,"outputs/final_model/soc_stock_modeling",target)
  if (identical(final_run_id,"latest")) {
    runs <- list.dirs(final_base,recursive=FALSE,full.names=FALSE)
    runs <- runs[grepl("^final_",runs)]
    if(!length(runs)) stop("No final model run under: ",final_base,call.=FALSE)
    final_run_id <- sort(runs,decreasing=TRUE)[1]
  }
  final_dir <- file.path(final_base,final_run_id)
  resumo <- readRDS(file.path(final_dir,"comparison/final_run_summary.rds"))
  if (identical(config_id,"auto")) config_id <- resumo$selected_cfgs$config_id[1]
  message("final_run_id: ",final_run_id," | config_id: ",config_id)

  tuning_id <- resumo$tuning_run_id
  tuning_dir <- file.path(project_root,"outputs/tuning/soc_stock_modeling",target,tuning_id)
  stopifnot(config_id %in% resumo$selected_cfgs$config_id)

  # ONE ROW PER CONFIG, AGGREGATED HERE.
  #
  # comparison_ranked.csv is one row per UNIT -- (config, fold, seed) -- and has
  # been since repetitions were introduced. This script reads it as one row per
  # config: it draws one bar per row, sizes points by parameter count, and
  # labelled the top three. Given 27 units it drew 27 bars for 3 configs and the
  # duplicate check at the top is what stopped it.
  #
  # The fix is not to drop the check. The metrics are AVERAGED over repetitions,
  # which is the number every decision in this project is read from; plotting a
  # single unit would plot one draw of the seed. The architecture fields are
  # identical within a config -- it is the same config -- so the first unit
  # supplies them.
  units <- ler(file.path(tuning_dir,"comparison/comparison_ranked.csv"))
  stopifnot(nrow(units) > 0L, "config_id" %in% names(units))
  num_mean <- function(d,col) if(col %in% names(d)) mean(as.numeric(d[[col]]),na.rm=TRUE) else NA_real_
  ranking <- do.call(rbind,lapply(split(units,units$config_id),function(d) {
    row <- d[1,,drop=FALSE]                       # architecture: same in every unit
    row$val_ccc     <- num_mean(d,"val_ccc")
    row$val_mae     <- num_mean(d,"val_mae")
    row$runtime_min <- num_mean(d,"runtime_min")
    row$n_units     <- nrow(d)
    row$unit_id     <- NULL                       # meaningless once averaged
    row
  }))
  ranking <- ranking[order(-ranking$val_ccc,ranking$val_mae),]
  ranking$rank <- seq_len(nrow(ranking))
  stopifnot(!anyDuplicated(ranking$config_id))
  message("configs: ",nrow(ranking)," (from ",nrow(units)," units)")
  cfg <- resumo$selected_cfgs[resumo$selected_cfgs$config_id==config_id,,drop=FALSE]
  seed_perf <- ler(file.path(final_dir,"comparison/all_seed_results_test.csv"))
  seed_perf <- seed_perf[seed_perf$config_id==config_id,,drop=FALSE]
  seed_perf <- seed_perf[order(seed_perf$seed),]
  seeds <- as.integer(seed_perf$seed)
  stopifnot(!anyDuplicated(seeds), setequal(seeds, resumo$seeds))
  pred <- lapply(seeds,function(s) ler(file.path(final_dir,config_id,"predictions",sprintf("seed%04d_pred_all.csv",s))))
  hist <- lapply(seeds,function(s) ler(file.path(final_dir,config_id,"history",sprintf("seed%04d_history.csv",s))))
  alinhar <- function(role) {
    dados <- lapply(pred,function(d) { d <- d[d$dataset_role==role,]; d[order(d$sample_id),] })
    ref <- dados[[1]]
    stopifnot(!anyDuplicated(ref$sample_id))
    for(d in dados) stopifnot(!anyDuplicated(d$sample_id), identical(d$sample_id,ref$sample_id),
      identical(as.character(d$profile_id),as.character(ref$profile_id)), isTRUE(all.equal(d$obs,ref$obs)))
    mat <- do.call(cbind,lapply(dados,`[[`,"pred"))
    stopifnot(all(is.finite(mat)),all(is.finite(ref$obs)))
    list(obs=ref$obs,pred=mat)
  }
  metricas <- function(y,z) {
    c(ccc=2*mean((y-mean(y))*(z-mean(z)))/(mean((y-mean(y))^2)+mean((z-mean(z))^2)+(mean(y)-mean(z))^2),
      mae=mean(abs(z-y)),rmse=sqrt(mean((z-y)^2)),r2=cor(y,z)^2,
      nse=1-sum((y-z)^2)/sum((y-mean(y))^2),bias=mean(z-y))
  }
  teste <- alinhar("test"); val <- alinhar("validation")
  y <- teste$obs; P <- teste$pred
  med <- apply(P,1,median); ens <- metricas(y,med)
  for(i in seq_along(seeds)) {
    m <- metricas(y,P[,i])
    for(nm in c("ccc","mae","rmse","r2","nse")) stopifnot(abs(m[nm]-seed_perf[[nm]][i])<2e-5)
  }
  vm <- as.data.frame(t(vapply(seq_along(seeds),function(i) metricas(val$obs,val$pred[,i]),numeric(6))))
  vm$seed <- seeds
  message("Computing every seed combination...")
  combinacoes <- lapply(seq_along(seeds),function(k) {
    ids <- combn(seq_along(seeds),k,simplify=FALSE)
    do.call(rbind,lapply(ids,function(ii) data.frame(k=k,seeds=paste(seeds[ii],collapse="|"),
      as.list(metricas(y,apply(P[,ii,drop=FALSE],1,median))),check.names=FALSE)))
  })
  comb <- do.call(rbind,combinacoes)
  stopifnot(nrow(comb)==2^length(seeds)-1)
  cortes <- quantile(y,c(0,.25,.5,.75,.9,.95,.99,1),names=FALSE)
  grupo <- vapply(y,function(v) sum(v>cortes[2:7])+1L,integer(1))
  rotulos <- c("0\u201325%","25\u201350%","50\u201375%","75\u201390%","90\u201395%","95\u201399%","99\u2013100%")
  faixas <- do.call(rbind,lapply(1:7,function(i) data.frame(grupo=rotulos[i],n=sum(grupo==i),
    inferior=cortes[i],superior=cortes[i+1],as.list(metricas(y[grupo==i],med[grupo==i])))))

  # Analytic count: convolutions, biases, BN affine parameters, SE, embedding and head.
  n_channels <- as.integer(readRDS(file.path(project_root,"outputs/patches/soc_stock_modeling",target,"patch_manifest.rds"))$n_channels[1])
  parametros <- function(v) {
    ch <- as.integer(strsplit(v$conv_channels,"_",fixed=TRUE)[[1]])
    ws <- as.integer(strsplit(as.character(v$window_sizes),"x",fixed=TRUE)[[1]])
    e <- v$embedding_dim; total <- 0
    pool <- if(is.null(v$embed_pool)) "flatten" else v$embed_pool
    for(w in ws) {
      anterior <- n_channels
      for(c in ch) {
        total <- total+anterior*c*9+c+2*c
        if(v$use_residual && anterior!=c) total <- total+anterior*c+c+2*c
        anterior <- c
      }
      c <- tail(ch,1)
      if(v$use_se_block) { h <- max(4,c%/%v$se_reduction); total <- total+c*h+h+h*c+c }
      total <- total+c*(if(pool=="gap") 1 else w*w)*e+3*e
    }
    if(length(ws)==2 && v$gate_type!="no_gate_concat") {
      g <- if(v$gate_type=="vector_featurewise") e else 1
      total <- total+4*e*e+3*e+e*g+g
    }
    total+e*length(ws)*e+3*e+e*128+384+128*64+64+65
  }
  ranking$parameters <- vapply(seq_len(nrow(ranking)),function(i) parametros(ranking[i,]),numeric(1))
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  ink <- "#183247"; ink_soft <- "#617486"; teal <- "#087f8c"; laranja <- "#e28b36"
  window_colours <- setNames(c("#087f8c","#e28b36","#7664a0","#4c78a8","#b45168","#579d68"),sort(unique(as.character(ranking$window_sizes))))
  seed_colours <- hcl.colors(length(seeds),"Viridis")
  figure_index <- list()
  figure_panel <- function(slug,heading,subheading,draw,caption,panels=1,height_in=7) {
    for(fmt in c("png","svg")) {
      out_file <- file.path(output_dir,paste0(slug,".",fmt))
      if(fmt=="png") png(out_file,width=14,height=height_in,units="in",res=190,bg="white")
      else svg(out_file,width=14,height=height_in,bg="white")
      tryCatch({
        par(mfrow=c(1,panels),oma=c(2,1,5,1),mar=c(5,5,2,1),family="sans",col=ink,
          col.axis=ink_soft,col.lab=ink,fg=ink,cex=1,las=1,bty="l",mgp=c(3,1,0))
        draw()
        mtext(heading,outer=TRUE,side=3,line=2.8,adj=0,cex=1.55,font=2,col=ink)
        mtext(subheading,outer=TRUE,side=3,line=1.2,adj=0,cex=.88,col=ink_soft)
        mtext(paste("SOC 0\u20135 cm |",config_id,"| Saved results \u2022 figures generated in R"),outer=TRUE,side=1,line=.7,adj=0,cex=.75,col=ink_soft)
      },finally=dev.off())
    }
    figure_index[[length(figure_index)+1L]] <<- list(slug=slug,heading=heading,caption=caption)
  }
  figure_panel("01_ranking","01 | Who got closest to the top?",
    sprintf("%d configs \u2022 validation CCC; MAE as tiebreaker",nrow(ranking)),function() {
      r <- ranking[nrow(ranking):1,]; yy <- seq_len(nrow(r))
      par(mar=c(5,7,2,1))
      plot(r$val_ccc,yy,type="n",xlim=c(0,max(r$val_ccc)+.07),yaxt="n",ylab="",xlab="Validation CCC -- higher is better")
      abline(v=seq(0,.7,.1),col="#edf1f4");segments(0,yy,r$val_ccc,yy,col="#dbe5e9",lwd=3)
      points(r$val_ccc,yy,pch=19,col=window_colours[as.character(r$window_sizes)],cex=1.25)
      axis(2,at=yy,labels=r$config_id,las=1,cex.axis=.85)
      text(r$val_ccc+.01,yy,sprintf("%.3f",r$val_ccc),adj=0,cex=.8)
      top <- which(r$rank<=3)
      text(.01,yy[top],paste(r$embed_pool[top],ifelse(r$gate_type[top]=="no_gate_concat","\u00b7 concat","\u00b7 vector gate")),adj=0,cex=.8)
      legend("bottomleft",legend=names(window_colours),col=window_colours,pch=19,bty="n",ncol=2,title="Windows (pixels)",cex=.85)
    },"One run per config. The small gap between the top-ranked configs does not establish statistical superiority.",height_in=10)
  figure_panel("02_desempenho_custo","02 | What performance fits your budget?",
    "Observed runtime \u00d7 validation CCC \u2022 circle area proportional to parameters",function() {
      plot(ranking$runtime_min,ranking$val_ccc,pch=21,bg=adjustcolor(window_colours[as.character(ranking$window_sizes)],alpha.f=.75),col="white",
        cex=sqrt(ranking$parameters/1e6)*.65,xlim=c(0,max(ranking$runtime_min)*1.16),ylim=range(ranking$val_ccc)+c(-.02,.02),
        xlab="Training time (min)",ylab="Validation CCC")
      f <- ranking[order(ranking$runtime_min),]; f <- f[f$val_ccc>c(-Inf,head(cummax(f$val_ccc),-1)),]
      lines(f$runtime_min,f$val_ccc,lty=2,col=teal)
      text(ranking$runtime_min[1:3],ranking$val_ccc[1:3],ranking$config_id[1:3],pos=c(3,1,3),cex=.85,font=2)
      legend("bottomright",legend=names(window_colours),col=window_colours,pch=19,bty="n",ncol=2,title="Windows (pixels)",cex=.8)
    },"Dashed line: observed efficiency frontier. Times depend on epochs, hardware and run conditions. Parameters counted from the current code's architecture.")
  figure_panel("03_aprendizado","03 | Learning trajectories by seed",
    "Each colour is a seed \u2022 circles mark the chosen checkpoints",function() {
      plot(NA,xlim=c(1,max(vapply(hist,function(h) max(h$epoch),numeric(1)))),ylim=range(unlist(lapply(hist,`[[`,"train_loss"))),log="y",xlab="Epoch",ylab="SmoothL1 on log1p (log scale)",main="Training \u2022 full trajectory")
      for(i in seq_along(seeds)) lines(hist[[i]]$epoch,hist[[i]]$train_loss,col=seed_colours[i])
      late <- unlist(lapply(hist,function(h) h$validation_loss[h$epoch>=10]))
      plot(NA,xlim=c(1,max(vapply(hist,function(h) max(h$epoch),numeric(1)))),ylim=c(min(late)-.005,quantile(late,.99)+.006),xlab="Epoch",ylab="SmoothL1 on log1p",main="Validation \u2022 convergence detail")
      for(i in seq_along(seeds)) {
        h <- hist[[i]]; lines(h$epoch,h$validation_loss,col=seed_colours[i]); b <- match(seed_perf$best_epoch[i],h$epoch)
        points(h$epoch[b],h$validation_loss[b],pch=21,bg=seed_colours[i],col="white",cex=1.1)
      }
      legend("topright",legend=seeds,col=seed_colours,lty=1,ncol=5,bty="n",cex=.6)
    },"Validation zoomed on the convergence region: early losses may fall off the axis. Training uses augmentation and dropout; validation runs in eval mode.",panels=2)
  original <- ranking[ranking$config_id==config_id,]
  figure_panel("04_estabilidade","04 | Does the winner repeat on new seeds?",
    "Validation \u2022 points = retrainings; bars = mean \u00b1 SD; diamond = original tuning",function() {
      for(m in c("ccc","mae","rmse")) {
        z <- vm[[m]]; orig <- original[[paste0("val_",m)]]; lim <- range(c(z,orig,mean(z)+c(-1,1)*sd(z)))
        plot(NA,xlim=c(-.3,1),ylim=lim+c(-1,1)*diff(lim)*.1,xaxt="n",xlab="",ylab="",main=switch(m,ccc="CCC (higher is better)",mae="MAE (t/ha)",rmse="RMSE (t/ha)"))
        abline(h=orig,col=laranja,lty=2); points(seq(-.15,.15,length.out=length(z)),z,pch=19,col=seed_colours)
        arrows(.45,mean(z)-sd(z),.45,mean(z)+sd(z),angle=90,code=3,length=.06,col=ink,lwd=2)
        points(.45,mean(z),pch=19);points(.8,orig,pch=18,col=laranja,cex=1.5)
        axis(1,at=c(0,.45,.8),labels=c("Seeds","Mean","Tuning"),cex.axis=.85)
      }
    },"Budget and stopping rules also differ between tuning and retraining. This comparison does not isolate the seed effect.",panels=3)
  figure_panel("05_ensemble_teste","05 | Does the ensemble hit where it matters?",
    sprintf("Test \u2022 %s profiles \u2022 median of %d seeds in native scale",format(length(y),big.mark=",",decimal.mark="."),length(seeds)),function() {
      lim <- max(y,med)*1.03
      smoothScatter(y,med,nrpoints=0,colramp=colorRampPalette(c("#ffffff","#92d2cb",teal,"#183247")),xlim=c(0,lim),ylim=c(0,lim),xlab="Observed (t/ha)",ylab="Predicted (t/ha)",asp=1)
      abline(0,1,lty=2,col=laranja,lwd=2)
      smoothScatter(y,med-y,nrpoints=0,colramp=colorRampPalette(c("#ffffff","#92d2cb",teal,"#183247")),xlab="Observed (t/ha)",ylab="Predicted \u2212 observed (t/ha)")
      abline(h=0,lty=2,col=laranja)
      legend("bottomleft",legend=c(sprintf("CCC: %.3f",ens["ccc"]),sprintf("MAE: %.2f t/ha",ens["mae"]),sprintf("RMSE: %.2f t/ha",ens["rmse"]),sprintf("Bias: %+.2f t/ha",ens["bias"])),bty="n",cex=.9)
    },"Darker colour means higher profile density (smoothed density). Negative residuals indicate underestimation. Metrics recomputed for the ensemble median.",panels=2)
  figure_panel("06_tamanho_ensemble","06 | How much does combining seeds gain?",
    sprintf("All %d combinations \u2022 line = median; band = 5\u201395 percentiles",nrow(comb)),function() {
      for(m in c("ccc","mae","rmse")) {
        q <- t(vapply(seq_along(seeds),function(k) quantile(comb[comb$k==k,m],c(.05,.5,.95)),numeric(3)))
        x <- seq_along(seeds)
        plot(x,q[,2],type="n",ylim=range(q),xlab="Number of seeds",ylab="",main=switch(m,ccc="CCC (higher is better)",mae="MAE (t/ha)",rmse="RMSE (t/ha)"))
        polygon(c(x,rev(x)),c(q[,1],rev(q[,3])),col=adjustcolor(teal,alpha.f=.18),border=NA)
        lines(x,q[,2],type="o",pch=19,col=teal,lwd=2);points(tail(x,1),ens[m],pch=19,col=laranja,cex=1.3)
      }
    },"Descriptive analysis on test, with architecture and seeds already fixed. The band measures variation across combinations; not a confidence interval. It does not pick the best subset.",panels=3)
  figure_panel("07_erros_por_faixa","07 | Where are the errors that matter most?",
    "Bands from the observed test quantiles \u2022 groups differ in size",function() {
      for(m in c("mae","bias")) {
        barplot(faixas[[m]],names.arg=paste0(rotulos,"\nn=",faixas$n),col=c(rep(teal,5),rep(laranja,2)),border=NA,
          cex.names=.68,ylab=if(m=="mae") "MAE (t/ha)" else "Predicted \u2212 observed (t/ha)",main=if(m=="mae") "Error magnitude" else "Error direction")
        abline(h=0,col=ink)
      }
    },"The last two bands highlight the largest stocks. Limits in t/ha and metrics are in the supporting table.",panels=2)
  write.csv2(comb,file.path(output_dir,"metricas_combinacoes.csv"),row.names=FALSE)
  write.csv2(faixas,file.path(output_dir,"metricas_faixas.csv"),row.names=FALSE)
  write.csv2(vm,file.path(output_dir,"metricas_seeds_validacao.csv"),row.names=FALSE)
  write.csv2(ranking,file.path(output_dir,"ranking_com_parametros.csv"),row.names=FALSE)
  write.csv2(data.frame(split="test",n=length(y),as.list(ens)),file.path(output_dir,"metricas_ensemble.csv"),row.names=FALSE)
  saveRDS(list(tuning_run_id=tuning_id,final_run_id=final_run_id,config_id=config_id,seeds=seeds,ensemble=ens),file.path(output_dir,"resumo.rds"))
  cards <- vapply(figure_index,function(f) paste0('<section><h2>',f$heading,'</h2><a href="',f$slug,'.png"><img src="',f$slug,'.png"></a><p>',f$caption,'</p><a href="',f$slug,'.svg">Editable SVG</a> \u00b7 <a href="',f$slug,'.png">PNG</a></section>'),character(1))
  pagina <- paste0('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>The CNN journey -- R</title><style>body{background:#edf2f5;color:#183247;font:17px/1.6 system-ui;margin:0}main{max-width:1150px;margin:auto;padding:40px 24px}h1{font-size:44px}section{background:white;padding:25px;border-radius:14px;margin:25px 0}img{width:100%}a{color:#087f8c}</style><main><h1>From selection to ensemble</h1><p>Figures generated entirely in R, from the saved results.</p><p>',config_id,' \u00b7 ',length(seeds),' seeds \u00b7 Ensemble CCC: ',sprintf('%.3f',ens['ccc']),' \u00b7 MAE: ',sprintf('%.2f',ens['mae']),' t/ha.</p>',paste(cards,collapse='\n'),'<p>Selection: validation. Ensemble: test. Stratified random split; not an independent spatial validation. Spread across seeds does not cover every source of uncertainty.</p><p>Tuning: ',tuning_id,'<br>Final: ',final_run_id,'</p></main></html>')
  writeLines(enc2utf8(pagina),file.path(output_dir,"galeria.html"),useBytes=TRUE)
  message("Figures written to: ",normalizePath(output_dir,winslash="/"))
  print(ens)
  invisible(list(ensemble=ens,output_dir=output_dir))
}

if (sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if(length(args)>=2) gerar_graficos_cnn(project_root=args[1],output_dir=args[2])
  else if(length(args)==1) gerar_graficos_cnn(project_root=args[1])
  else gerar_graficos_cnn()
} else {
  message("Function loaded. Run gerar_graficos_cnn() to generate inside the project, or pass output_dir.")
}
