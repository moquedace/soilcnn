# Gerar os tres mapas de fechamento quando a predicao FULL estiver concluida.
# Preencha os caminhos das bandas de UM MESMO modelo, fonte de calibracao e area.
mediana <- "PREENCHER/ensemble_median.vrt"
limite_inferior <- "PREENCHER/pi90_constant_lower_block.vrt"
limite_superior <- "PREENCHER/pi90_constant_upper_block.vrt"
area_aplicabilidade <- "PREENCHER/aoa_block.vrt"
arquivo_saida <- "PREENCHER/prediction_interval_aoa.png"
# Recorte regional opcional em coordenadas do raster: c(xmin,xmax,ymin,ymax).
extensao <- NULL
titulo_propriedade <- "SOC stock, 0-30 cm (t/ha)"
concluido_full <- FALSE  # Mude para TRUE somente depois de conferir a conclusao do full.

if (!concluido_full) stop("A figura exige uma predicao full concluida. Confira o run antes de habilitar.")
if (!requireNamespace("terra",quietly=TRUE)) stop("Instale terra: install.packages('terra')")
caminhos<-c(mediana,limite_inferior,limite_superior,area_aplicabilidade)
if(any(grepl("PREENCHER",c(caminhos,arquivo_saida))))stop("Preencha os caminhos no topo.")
if(any(!file.exists(caminhos)))stop("Alguma banda nao foi encontrada.")
r<-lapply(caminhos,terra::rast)
for(i in 2:4)if(!terra::compareGeom(r[[1]],r[[i]],stopOnError=FALSE))stop("Bandas desalinhadas.")
if(!is.null(extensao))r<-lapply(r,function(x)terra::crop(x,terra::ext(extensao)))
# Faca as operacoes antes de reduzir para exibicao. O raster original fica intacto.
largura<-r[[3]]-r[[2]]
if(terra::global(largura<0,"sum",na.rm=TRUE)[1,1]>0)stop("Intervalo com limite superior menor que inferior.")
# Projecao cartografica em metros: a proporcao da imagem nao altera os mapas.
crs_lac<-"+proj=laea +lat_0=-15 +lon_0=-75 +datum=WGS84 +units=m +no_defs"
overview<-terra::spatSample(r[[1]],250000,method="regular",as.raster=TRUE)
template<-terra::project(overview,crs_lac)
cartografia<-list(terra::project(overview,template,method="bilinear"),
 terra::project(terra::spatSample(largura,250000,method="regular",as.raster=TRUE),template,method="bilinear"),
 terra::project(terra::spatSample(r[[4]],250000,method="regular",as.raster=TRUE),template,method="near"))
dir.create(dirname(arquivo_saida),recursive=TRUE,showWarnings=FALSE)
png(arquivo_saida,width=2400,height=1050,res=200)
par(mfrow=c(1,3),mar=c(2,1,4,5),family="sans",col="#253840",col.main="#253840")
terra::plot(cartografia[[1]],main=titulo_propriedade,axes=FALSE,col=hcl.colors(80,"Blues 3"),asp=1)
terra::plot(cartografia[[2]],main="90% prediction interval width (t/ha)",axes=FALSE,col=hcl.colors(80,"YlOrBr"),asp=1)
terra::plot(cartografia[[3]],main="Area of applicability",axes=FALSE,col=c("#DADFDA","#60734F"),
            breaks=c(-.5,.5,1.5),legend=FALSE,asp=1)
legend("bottomleft",legend=c("Outside","Inside"),fill=c("#DADFDA","#60734F"),bty="n",cex=.85)
dev.off()
message("Figura salva em ",arquivo_saida)
