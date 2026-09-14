# Avalia\u00e7\u00e3o gr\u00e1fica da CNN \u2014 apenas R base, sem instalar pacotes.
# RStudio: source("examples/soc_stock_0_5cm/06_avaliacao_grafica.R", encoding="UTF-8")
# Console: Rscript gerar_graficos.R "PASTA_PROJETO" "PASTA_SAIDA"
# N\u00e3o treina modelos nem modifica os resultados de origem.

gerar_graficos_cnn <- function(
  project_root = "D:/usuario_armazenamento/cassio/R/deep_learning_caret",
  output_dir = file.path(project_root, "outputs", "avaliacao_grafica"),
  final_run_id = "final_20260823_001856",
  config_id = "cfg_014"
) {
  if (.Platform$OS.type == "windows") {
    local_antigo <- Sys.getlocale("LC_CTYPE")
    suppressWarnings(Sys.setlocale("LC_CTYPE", "Portuguese_Brazil.utf8"))
    on.exit(suppressWarnings(Sys.setlocale("LC_CTYPE", local_antigo)), add=TRUE)
  }
  ler <- function(f) read.csv2(f, stringsAsFactors=FALSE, check.names=FALSE)
  target <- "soc_stock_0_5cm"
  final_dir <- file.path(project_root,"outputs/final_model/soc_stock_modeling",target,final_run_id)
  resumo <- readRDS(file.path(final_dir,"comparison/final_run_summary.rds"))
  tuning_id <- resumo$tuning_run_id
  tuning_dir <- file.path(project_root,"outputs/tuning/soc_stock_modeling",target,tuning_id)
  ranking <- ler(file.path(tuning_dir,"comparison/comparison_ranked.csv"))
  ranking <- ranking[order(ranking$rank),]
  stopifnot(config_id %in% resumo$selected_cfgs$config_id, !anyDuplicated(ranking$config_id))
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
  message("Calculando todas as combina\u00e7\u00f5es de sementes...")
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

  # Contagem anal\u00edtica: convolu\u00e7\u00f5es, vieses, par\u00e2metros afins BN, SE, embedding e head.
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
  tinta <- "#183247"; suave <- "#617486"; teal <- "#087f8c"; laranja <- "#e28b36"
  cores <- setNames(c("#087f8c","#e28b36","#7664a0","#4c78a8","#b45168","#579d68"),sort(unique(as.character(ranking$window_sizes))))
  seed_cores <- hcl.colors(length(seeds),"Viridis")
  figuras <- list()
  figura <- function(nome,titulo,subtitulo,desenhar,legenda,paineis=1,altura=7) {
    for(tipo in c("png","svg")) {
      destino <- file.path(output_dir,paste0(nome,".",tipo))
      if(tipo=="png") png(destino,width=14,height=altura,units="in",res=190,bg="white")
      else svg(destino,width=14,height=altura,bg="white")
      tryCatch({
        par(mfrow=c(1,paineis),oma=c(2,1,5,1),mar=c(5,5,2,1),family="sans",col=tinta,
          col.axis=suave,col.lab=tinta,fg=tinta,cex=1,las=1,bty="l",mgp=c(3,1,0))
        desenhar()
        mtext(titulo,outer=TRUE,side=3,line=2.8,adj=0,cex=1.55,font=2,col=tinta)
        mtext(subtitulo,outer=TRUE,side=3,line=1.2,adj=0,cex=.88,col=suave)
        mtext(paste("SOC 0\u20135 cm |",config_id,"| Resultados salvos \u2022 gr\u00e1ficos gerados em R"),outer=TRUE,side=1,line=.7,adj=0,cex=.75,col=suave)
      },finally=dev.off())
    }
    figuras[[length(figuras)+1L]] <<- list(nome=nome,titulo=titulo,legenda=legenda)
  }
  figura("01_ranking","01 | Quem chegou mais perto do topo?",
    sprintf("%d configura\u00e7\u00f5es \u2022 CCC de valida\u00e7\u00e3o; MAE como desempate",nrow(ranking)),function() {
      r <- ranking[nrow(ranking):1,]; yy <- seq_len(nrow(r))
      par(mar=c(5,7,2,1))
      plot(r$val_ccc,yy,type="n",xlim=c(0,max(r$val_ccc)+.07),yaxt="n",ylab="",xlab="CCC de valida\u00e7\u00e3o \u2014 maior \u00e9 melhor")
      abline(v=seq(0,.7,.1),col="#edf1f4");segments(0,yy,r$val_ccc,yy,col="#dbe5e9",lwd=3)
      points(r$val_ccc,yy,pch=19,col=cores[as.character(r$window_sizes)],cex=1.25)
      axis(2,at=yy,labels=r$config_id,las=1,cex.axis=.85)
      text(r$val_ccc+.01,yy,sprintf("%.3f",r$val_ccc),adj=0,cex=.8)
      top <- which(r$rank<=3)
      text(.01,yy[top],paste(r$embed_pool[top],ifelse(r$gate_type[top]=="no_gate_concat","\u00b7 concatena\u00e7\u00e3o","\u00b7 gate vetorial")),adj=0,cex=.8)
      legend("bottomleft",legend=names(cores),col=cores,pch=19,bty="n",ncol=2,title="Janelas (pixels)",cex=.85)
    },"Uma execu\u00e7\u00e3o por configura\u00e7\u00e3o. A pequena diferen\u00e7a entre os primeiros colocados n\u00e3o estabelece superioridade estat\u00edstica.",altura=10)
  figura("02_desempenho_custo","02 | Qual desempenho cabe no seu or\u00e7amento?",
    "Tempo observado \u00d7 CCC de valida\u00e7\u00e3o \u2022 \u00e1rea dos c\u00edrculos proporcional aos par\u00e2metros",function() {
      plot(ranking$runtime_min,ranking$val_ccc,pch=21,bg=adjustcolor(cores[as.character(ranking$window_sizes)],alpha.f=.75),col="white",
        cex=sqrt(ranking$parameters/1e6)*.65,xlim=c(0,max(ranking$runtime_min)*1.16),ylim=range(ranking$val_ccc)+c(-.02,.02),
        xlab="Tempo de treinamento (min)",ylab="CCC de valida\u00e7\u00e3o")
      f <- ranking[order(ranking$runtime_min),]; f <- f[f$val_ccc>c(-Inf,head(cummax(f$val_ccc),-1)),]
      lines(f$runtime_min,f$val_ccc,lty=2,col=teal)
      text(ranking$runtime_min[1:3],ranking$val_ccc[1:3],ranking$config_id[1:3],pos=c(3,1,3),cex=.85,font=2)
      legend("bottomright",legend=names(cores),col=cores,pch=19,bty="n",ncol=2,title="Janelas (pixels)",cex=.8)
    },"Linha tracejada: fronteira de efici\u00eancia observada. Tempos dependem de \u00e9pocas, hardware e condi\u00e7\u00f5es da execu\u00e7\u00e3o. Par\u00e2metros contados a partir da arquitetura do c\u00f3digo atual.")
  figura("03_aprendizado","03 | Trajet\u00f3rias de aprendizado por semente",
    "Cada cor \u00e9 uma semente \u2022 c\u00edrculos marcam os checkpoints escolhidos",function() {
      plot(NA,xlim=c(1,max(vapply(hist,function(h) max(h$epoch),numeric(1)))),ylim=range(unlist(lapply(hist,`[[`,"train_loss"))),log="y",xlab="\u00c9poca",ylab="SmoothL1 em log1p (escala log)",main="Treino \u2022 trajet\u00f3ria completa")
      for(i in seq_along(seeds)) lines(hist[[i]]$epoch,hist[[i]]$train_loss,col=seed_cores[i])
      late <- unlist(lapply(hist,function(h) h$validation_loss[h$epoch>=10]))
      plot(NA,xlim=c(1,max(vapply(hist,function(h) max(h$epoch),numeric(1)))),ylim=c(min(late)-.005,quantile(late,.99)+.006),xlab="\u00c9poca",ylab="SmoothL1 em log1p",main="Valida\u00e7\u00e3o \u2022 detalhe da converg\u00eancia")
      for(i in seq_along(seeds)) {
        h <- hist[[i]]; lines(h$epoch,h$validation_loss,col=seed_cores[i]); b <- match(seed_perf$best_epoch[i],h$epoch)
        points(h$epoch[b],h$validation_loss[b],pch=21,bg=seed_cores[i],col="white",cex=1.1)
      }
      legend("topright",legend=seeds,col=seed_cores,lty=1,ncol=5,bty="n",cex=.6)
    },"Valida\u00e7\u00e3o ampliada na regi\u00e3o de converg\u00eancia: perdas iniciais podem ficar fora do eixo. Treino inclui augmentation e dropout; valida\u00e7\u00e3o usa modo de avalia\u00e7\u00e3o.",paineis=2)
  original <- ranking[ranking$config_id==config_id,]
  figura("04_estabilidade","04 | O vencedor se repete em novas sementes?",
    "Valida\u00e7\u00e3o \u2022 pontos = retreinamentos; barras = m\u00e9dia \u00b1 DP; losango = tuning original",function() {
      for(m in c("ccc","mae","rmse")) {
        z <- vm[[m]]; orig <- original[[paste0("val_",m)]]; lim <- range(c(z,orig,mean(z)+c(-1,1)*sd(z)))
        plot(NA,xlim=c(-.3,1),ylim=lim+c(-1,1)*diff(lim)*.1,xaxt="n",xlab="",ylab="",main=switch(m,ccc="CCC (maior \u00e9 melhor)",mae="MAE (t/ha)",rmse="RMSE (t/ha)"))
        abline(h=orig,col=laranja,lty=2); points(seq(-.15,.15,length.out=length(z)),z,pch=19,col=seed_cores)
        arrows(.45,mean(z)-sd(z),.45,mean(z)+sd(z),angle=90,code=3,length=.06,col=tinta,lwd=2)
        points(.45,mean(z),pch=19);points(.8,orig,pch=18,col=laranja,cex=1.5)
        axis(1,at=c(0,.45,.8),labels=c("Seeds","M\u00e9dia","Tuning"),cex.axis=.85)
      }
    },"O or\u00e7amento e as regras de parada tamb\u00e9m diferem entre tuning e retreinamento. Esta compara\u00e7\u00e3o n\u00e3o isola o efeito da semente.",paineis=3)
  figura("05_ensemble_teste","05 | O ensemble acerta onde importa?",
    sprintf("Teste \u2022 %s perfis \u2022 mediana de %d sementes em escala nativa",format(length(y),big.mark=".",decimal.mark=","),length(seeds)),function() {
      lim <- max(y,med)*1.03
      smoothScatter(y,med,nrpoints=0,colramp=colorRampPalette(c("#ffffff","#92d2cb",teal,"#183247")),xlim=c(0,lim),ylim=c(0,lim),xlab="Observado (t/ha)",ylab="Predito (t/ha)",asp=1)
      abline(0,1,lty=2,col=laranja,lwd=2)
      smoothScatter(y,med-y,nrpoints=0,colramp=colorRampPalette(c("#ffffff","#92d2cb",teal,"#183247")),xlab="Observado (t/ha)",ylab="Predito \u2212 observado (t/ha)")
      abline(h=0,lty=2,col=laranja)
      legend("bottomleft",legend=c(sprintf("CCC: %.3f",ens["ccc"]),sprintf("MAE: %.2f t/ha",ens["mae"]),sprintf("RMSE: %.2f t/ha",ens["rmse"]),sprintf("Vi\u00e9s: %+.2f t/ha",ens["bias"])),bty="n",cex=.9)
    },"Cor mais escura indica maior densidade de perfis (densidade suavizada). Res\u00edduos negativos indicam subestima\u00e7\u00e3o. M\u00e9tricas recalculadas para a mediana do ensemble.",paineis=2)
  figura("06_tamanho_ensemble","06 | Quanto se ganha ao combinar sementes?",
    sprintf("Todas as %d combina\u00e7\u00f5es \u2022 linha = mediana; faixa = percentis 5\u201395",nrow(comb)),function() {
      for(m in c("ccc","mae","rmse")) {
        q <- t(vapply(seq_along(seeds),function(k) quantile(comb[comb$k==k,m],c(.05,.5,.95)),numeric(3)))
        x <- seq_along(seeds)
        plot(x,q[,2],type="n",ylim=range(q),xlab="N\u00famero de sementes",ylab="",main=switch(m,ccc="CCC (maior \u00e9 melhor)",mae="MAE (t/ha)",rmse="RMSE (t/ha)"))
        polygon(c(x,rev(x)),c(q[,1],rev(q[,3])),col=adjustcolor(teal,alpha.f=.18),border=NA)
        lines(x,q[,2],type="o",pch=19,col=teal,lwd=2);points(tail(x,1),ens[m],pch=19,col=laranja,cex=1.3)
      }
    },"An\u00e1lise descritiva no teste com arquitetura e sementes j\u00e1 fixadas. A faixa mede varia\u00e7\u00e3o entre combina\u00e7\u00f5es; n\u00e3o \u00e9 intervalo de confian\u00e7a. N\u00e3o seleciona o melhor subconjunto.",paineis=3)
  figura("07_erros_por_faixa","07 | Onde est\u00e3o os erros mais importantes?",
    "Faixas definidas pelos quantis observados do teste \u2022 grupos t\u00eam tamanhos diferentes",function() {
      for(m in c("mae","bias")) {
        barplot(faixas[[m]],names.arg=paste0(rotulos,"\nn=",faixas$n),col=c(rep(teal,5),rep(laranja,2)),border=NA,
          cex.names=.68,ylab=if(m=="mae") "MAE (t/ha)" else "Predito \u2212 observado (t/ha)",main=if(m=="mae") "Magnitude do erro" else "Dire\u00e7\u00e3o do erro")
        abline(h=0,col=tinta)
      }
    },"As duas \u00faltimas faixas destacam os maiores estoques. Limites em t/ha e m\u00e9tricas constam na tabela de apoio.",paineis=2)
  write.csv2(comb,file.path(output_dir,"metricas_combinacoes.csv"),row.names=FALSE)
  write.csv2(faixas,file.path(output_dir,"metricas_faixas.csv"),row.names=FALSE)
  write.csv2(vm,file.path(output_dir,"metricas_seeds_validacao.csv"),row.names=FALSE)
  write.csv2(ranking,file.path(output_dir,"ranking_com_parametros.csv"),row.names=FALSE)
  write.csv2(data.frame(split="test",n=length(y),as.list(ens)),file.path(output_dir,"metricas_ensemble.csv"),row.names=FALSE)
  saveRDS(list(tuning_run_id=tuning_id,final_run_id=final_run_id,config_id=config_id,seeds=seeds,ensemble=ens),file.path(output_dir,"resumo.rds"))
  cards <- vapply(figuras,function(f) paste0('<section><h2>',f$titulo,'</h2><a href="',f$nome,'.png"><img src="',f$nome,'.png"></a><p>',f$legenda,'</p><a href="',f$nome,'.svg">SVG edit\u00e1vel</a> \u00b7 <a href="',f$nome,'.png">PNG</a></section>'),character(1))
  pagina <- paste0('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A jornada da CNN \u2014 R</title><style>body{background:#edf2f5;color:#183247;font:17px/1.6 system-ui;margin:0}main{max-width:1150px;margin:auto;padding:40px 24px}h1{font-size:44px}section{background:white;padding:25px;border-radius:14px;margin:25px 0}img{width:100%}a{color:#087f8c}</style><main><h1>Da sele\u00e7\u00e3o ao ensemble</h1><p>Gr\u00e1ficos gerados integralmente em R, a partir dos resultados salvos.</p><p>',config_id,' \u00b7 ',length(seeds),' sementes \u00b7 CCC do ensemble: ',sprintf('%.3f',ens['ccc']),' \u00b7 MAE: ',sprintf('%.2f',ens['mae']),' t/ha.</p>',paste(cards,collapse='\n'),'<p>Sele\u00e7\u00e3o: valida\u00e7\u00e3o. Ensemble: teste. Split aleat\u00f3rio estratificado; n\u00e3o representa valida\u00e7\u00e3o espacial independente. Dispers\u00e3o entre sementes n\u00e3o cobre todas as fontes de incerteza.</p><p>Tuning: ',tuning_id,'<br>Final: ',final_run_id,'</p></main></html>')
  writeLines(enc2utf8(pagina),file.path(output_dir,"galeria.html"),useBytes=TRUE)
  message("Figuras geradas em: ",normalizePath(output_dir,winslash="/"))
  print(ens)
  invisible(list(ensemble=ens,output_dir=output_dir))
}

if (sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if(length(args)>=2) gerar_graficos_cnn(project_root=args[1],output_dir=args[2])
  else if(length(args)==1) gerar_graficos_cnn(project_root=args[1])
  else gerar_graficos_cnn()
} else {
  message("Fun\u00e7\u00e3o carregada. Execute gerar_graficos_cnn() para gerar no projeto, ou informe output_dir.")
}
