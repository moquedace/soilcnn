# soilcnn editorial figures. No fitting or prediction is performed.
# Base graphics keep this authoring script independent of torch and model state.
# Raster reading uses terra, or a locally installed GDAL command-line fallback.
# Override these paths before source() when using another application dataset.
if (!exists("root")) root <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
if (!exists("trial")) trial <- "D:/usuario_armazenamento/cassio/projects/soc_stock_0_30cm_lac/outputs"
if (!exists("rdir")) rdir <- "D:/usuario_armazenamento/cassio/data/predictors_resolution_250m_latam"
fig <- file.path(root, "vignettes", "figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)
# Where each figure came from: a record for the author, kept beside the
# figure inputs and out of vignettes/, which is installed with the package.
records <- file.path(root, "tools", "figure_inputs")
dir.create(records, recursive = TRUE, showWarnings = FALSE)
csv <- function(p) read.csv2(p, stringsAsFactors = FALSE, check.names = FALSE)
meta <- csv(file.path(trial, "full/patches/patch_meta.csv"))
C <- c(ink="#253840", muted="#63767D", blue="#246589", olive="#60734F",
       earth="#AB6645", pale="#F3F6F5", line="#DCE4E3", train="#A9B6BA")
if (.Platform$OS.type == "windows") windowsFonts(editorial = windowsFont("Segoe UI"))
font <- if (.Platform$OS.type == "windows") "editorial" else "sans"
start <- function(name, w=1800, h=1100) {
  png(file.path(fig,paste0(name,".png")),w,h,res=180,type=if(.Platform$OS.type=="windows") "windows" else "cairo")
  par(mar=rep(0,4),family=font,fg=C["ink"],col=C["ink"],xpd=NA)
  plot.new(); plot.window(c(0,1),c(0,1),xaxs="i",yaxs="i")
}
txt <- function(x,y,s,size=1,col=C["ink"],bold=FALSE,adj=0) text(x,y,s,cex=size,col=col,font=if(bold) 2 else 1,adj=adj)
# The kicker names the figure's subject; the vignette's captions number them.
heading <- function(k,title,sub) {
  k <- sub("^[0-9]+ / ","",k)
  txt(.04,.953,k,.76,C["blue"],TRUE); txt(.04,.902,title,1.65,bold=TRUE)
  txt(.04,.856,sub,.86,C["muted"])
}
box <- function(x0,y0,x1,y1,col=C["pale"],border=NA) rect(x0,y0,x1,y1,col=col,border=border)
arrow <- function(x0,y0,x1,y1) arrows(x0,y0,x1,y1,length=.07,col=C["muted"],lwd=1.2)
finish <- function() invisible(dev.off())

# Read only small raster windows or an overview, never the full raster stack.
read_raster <- function(nm,win=NULL,overview=FALSE) {
  f <- file.path(rdir,paste0(nm,".tif"))
  if (requireNamespace("terra",quietly=TRUE)) {
    r <- terra::rast(f)
    if (!is.null(win)) {
      e <- terra::ext(origin[1]+win[1]*cell,origin[1]+(win[1]+win[3])*cell,
                      origin[2]-(win[2]+win[4])*cell,origin[2]-win[2]*cell)
      r <- terra::crop(r,e)
    }
    if (overview) r <- terra::spatSample(r,200000,method="regular",as.raster=TRUE)
    d <- as.data.frame(r,xy=TRUE,na.rm=FALSE); names(d)<-c("x","y","z"); return(d)
  }
  exe <- Sys.which("gdal_translate")
  if (!nzchar(exe) && file.exists("C:/OSGeo4W/bin/gdal_translate.exe")) exe <- "C:/OSGeo4W/bin/gdal_translate.exe"
  if (!nzchar(exe)) stop("Figure authoring needs terra or gdal_translate; no model training is required.")
  p <- tempfile(fileext=".xyz"); on.exit(unlink(p))
  a <- c("-q","-of","XYZ")
  if (!is.null(win)) a <- c(a,"-srcwin",as.character(win))
  if (overview) a <- c(a,"-outsize","440","455","-r","nearest")
  status <- system2(exe,c(a,shQuote(f),shQuote(p)),stdout=FALSE,stderr=FALSE)
  if (status != 0) stop("GDAL could not read ",f)
  d <- read.table(p,col.names=c("x","y","z")); d$z[abs(d$z)>1e15]<-NA; d
}
# Pixel geometry is read from the dataset when terra is available. The fallback
# uses the checked application's documented grid, not an assumed metric CRS.
origin <- c(-118.5010626473789,33.00145808279274); cell <- .0022457981117329
if (requireNamespace("terra",quietly=TRUE)) {
  rr<-terra::rast(file.path(rdir,"landsat_2020_2025_ndvi.tif"))
  origin<-c(terra::xmin(rr),terra::ymax(rr)); cell<-terra::res(rr)[1]
}
here <- which.min((meta$x+47.65)^2+(meta$y+22.72)^2)
col0 <- floor((meta$x[here]-origin[1])/cell); row0 <- floor((origin[2]-meta$y[here])/cell)
layers <- c("landsat_2020_2025_ndvi","ensemble_digital_terrain_model_v1_1","clay")
pal <- list(colorRampPalette(c("#F0F3DD","#A4B58C","#60734F","#2C4F3F"))(80),
            colorRampPalette(c("#F0F4F6","#A9C5D2","#588BA5","#244F69"))(80),
            colorRampPalette(c("#FBF0DA","#DAB890","#AF7850","#70442D"))(80))
ras <- lapply(layers,read_raster,win=c(col0-12,row0-12,25,25))
limits <- lapply(ras,function(d) range(d$z,na.rm=TRUE))
tile <- function(d,x0,y0,x1,y1,palette,zlim) {
  xs<-sort(unique(d$x)); ys<-sort(unique(d$y),decreasing=TRUE)
  ix<-match(d$x,xs); iy<-match(d$y,ys)
  ci<-pmax(1,pmin(length(palette),1+floor((d$z-zlim[1])/diff(zlim)*(length(palette)-1))))
  colors<-palette[ci]; colors[is.na(d$z)]<-"#F0F2F1"
  dx<-(x1-x0)/length(xs); dy<-(y1-y0)/length(ys)
  rect(x0+(ix-1)*dx,y1-iy*dy,x0+ix*dx,y1-(iy-1)*dy,col=colors,border=NA)
}

start("workflow",1800,1000)
heading("01 / WORKFLOW","From soil observations to a defensible map","One shared dataset. An explicit validation design. Predictions with calibrated intervals and applicability.")
verbs<-c("Prepare","Load","Validate","Tune","Refit","Predict")
calls<-c("dsm_prepare()","dsm_load()","spatial_cv() + alternatives","dsm_train()","dsm_final()","dsm_predict()")
detail<-c("Profiles + aligned rasters\nRaw multiscale patches","Store + extraction recipe\nConsistency checks","Train / validate / calibrate / test\nSpatial separation","Configurations x folds x seeds\nSelection + model baselines","Selected configuration\nEnsemble + calibration","Property + interval bands\nDissimilarity + applicability")
for(i in 1:6) {
  j<-(i-1)%%3; row<-(i-1)%/%3; x<-.04+j*.315; y<-.49-row*.34
  box(x,y,x+.285,y+.27,if(i==3) "#F7F0E9" else C["pale"])
  txt(x+.018,y+.225,sprintf("%02d",i),.85,if(i==3) C["earth"] else C["blue"],TRUE)
  txt(x+.057,y+.225,verbs[i],1.18,bold=TRUE)
  txt(x+.018,y+.165,calls[i],.76,C["blue"])
  txt(x+.018,y+.083,detail[i],.83,C["muted"])
  if(j<2) arrow(x+.292,y+.135,x+.309,y+.135)
}
segments(.958,.625,.976,.625,col=C["muted"]); segments(.976,.625,.976,.465,col=C["muted"])
segments(.976,.465,.025,.465,col=C["muted"]); segments(.025,.465,.025,.285,col=C["muted"]); arrow(.025,.285,.038,.285)
box(.04,.027,.955,.115,C["pale"])
txt(.058,.083,"07  Interpret",1.05,bold=TRUE)
txt(.25,.083,"dsm_importance() + importance_map()",.85,C["blue"])
txt(.058,.049,"Theme rankings, neighbourhood context and spatial effects; read with uncertainty and applicability.",.77,C["muted"])
arrow(.812,.145,.812,.118)
finish()

start("patches",1800,1620)
heading("02 / MODEL INPUT","One location, several channels, three spatial scales","Real raster cells around one soil profile in southeastern Brazil. Colours retain their own scale within each channel.")
titles<-c("Vegetation / NDVI","Terrain / elevation","Soil / clay")
for(j in 1:3) {
 x<-.15+(j-1)*.285; txt(x,.799,titles[j],1.08,bold=TRUE)
 for(i in 1:3) {
  w<-c(3,9,15)[i]; h<-(w-1)/2
  d<-ras[[j]]; xx<-sort(unique(d$x)); yy<-sort(unique(d$y),decreasing=TRUE)
  d<-d[d$x %in% xx[(13-h):(13+h)] & d$y %in% yy[(13-h):(13+h)],]
  y<-.544-(i-1)*.216
  tile(d,x,y,x+.205,y+.205,pal[[j]],limits[[j]])
  # Mark the central raster cell, not the exact subcell observation position.
  points(x+.1025,y+.1025,pch=21,bg="white",col=C["ink"],cex=.6,lwd=1)
 }
 for(k in seq_along(pal[[j]])) rect(x+(k-1)*.205/80,.061,x+k*.205/80,.074,col=pal[[j]][k],border=NA)
 txt(x,.043,format(round(limits[[j]][1],2),trim=TRUE),.65,C["muted"])
 txt(x+.205,.043,format(round(limits[[j]][2],2),trim=TRUE),.65,C["muted"],adj=1)
}
for(i in 1:3) {
 y<-.646-(i-1)*.216; txt(.035,y+.015,paste0(c(3,9,15)[i]," x ",c(3,9,15)[i]),.9,bold=TRUE)
 txt(.035,y-.015,paste0(c("0.75","2.25","3.75")[i]," km*"),.72,C["muted"])
}
txt(.15,.015,"* Nominal widths at 250 m. This longitude/latitude grid has latitude-dependent ground dimensions. Units are source raster values.",.63,C["muted"])
finish()

start("architecture",1800,950)
heading("03 / ARCHITECTURE","Learn from the centre and its neighbourhood","A conceptual two-branch configuration. The tuning grid also supports a single spatial scale.")
for(i in 1:2) {
 y<-c(.54,.23)[i]; x<-.055
 for(k in 3:1) box(x+k*.012,y+k*.013,x+.105+k*.012,y+.15+k*.013,c("#C9DAE3","#ABC4D2","#86ACBF")[k],"white")
 txt(.055,y-.027,if(i==1) "Small patch" else "Large patch",.9,bold=TRUE)
 arrow(.21,y+.09,.265,y+.09)
 box(.28,y+.025,.49,y+.16)
 txt(.385,y+.11,"Convolution blocks",.94,bold=TRUE,adj=.5)
 txt(.385,y+.065,"Spatial representation",.8,C["muted"],adj=.5)
 arrow(.50,y+.09,.61,.445)
}
box(.625,.355,.765,.535,"#EAF0E6")
txt(.695,.468,"Learned gate",1,bold=TRUE,adj=.5)
txt(.695,.411,"Fuse the scales",.78,C["olive"],adj=.5)
arrow(.777,.445,.818,.445); box(.835,.355,.965,.535,"#EAF0F5")
txt(.9,.468,"Prediction",1,bold=TRUE,adj=.5); txt(.9,.411,"Soil property",.8,C["blue"],adj=.5)
txt(.04,.085,"Each patch is a stack of predictor channels, scaled with constants fitted on the training rows of each fold.",.85,C["muted"])
finish()

# Geographic overview from a downsampled predictor validity mask.
land <- read_raster("bio1",overview=TRUE)
land<-land[is.finite(land$z) & land$z > -9990,]
# Spherical Lambert azimuthal equal-area, centred on Latin America.
# Only illustration coordinates are projected; training rasters and folds stay intact.
project_lac <- function(lon,lat) {
  rad <- pi/180; lambda <- (lon+75)*rad; phi <- lat*rad; phi0 <- -15*rad
  denom <- 1+sin(phi0)*sin(phi)+cos(phi0)*cos(phi)*cos(lambda)
  if(any(denom<=0))stop("Projection contains an antipodal point")
  k <- sqrt(2/denom); radius <- 6371008.8
  cbind(x=radius*k*cos(phi)*sin(lambda),
        y=radius*k*(cos(phi0)*sin(phi)-sin(phi0)*cos(phi)*cos(lambda)))
}
land_xy <- project_lac(land$x,land$y)
map <- function(x0,y0,x1,y1,points_data,roles=NULL) {
  xy <- project_lac(points_data$x,points_data$y)
  xr<-range(land_xy[,1]);yr<-range(land_xy[,2])
  # Fit coordinates in physical device inches, with exactly equal x/y scales.
  # Normalised figure coordinates alone do not preserve aspect on a rectangular PNG.
  inches<-par("pin")
  scale<-min((x1-x0)*inches[1]/diff(xr),(y1-y0)*inches[2]/diff(yr))*.98
  sx<-scale/inches[1];sy<-scale/inches[2]
  ox<-(x0+x1)/2-mean(xr)*sx;oy<-(y0+y1)/2-mean(yr)*sy
  px<-function(v)ox+v*sx;py<-function(v)oy+v*sy
  stopifnot(abs(sx*inches[1]-sy*inches[2])<1e-12)
  points(px(land_xy[,1]),py(land_xy[,2]),pch=15,cex=.31,col="#E9EEEC")
  if(is.null(roles)) points(px(xy[,1]),py(xy[,2]),pch=16,cex=.22,col=adjustcolor(C["blue"],.3))
  else {
   for(role in c("train","out","test","validation")) {
    q<-which(roles==role); co<-c(train=unname(C["train"]),out=unname(C["muted"]),test=unname(C["earth"]),validation=unname(C["blue"]))[role]
    points(px(xy[q,1]),py(xy[q,2]),pch=if(role=="out") 1 else if(role=="test") 17 else 16,
           cex=if(role=="train") .38 else .65,col=co)
   }
  }
}
start("observations",1800,1250)
heading("04 / APPLICATION DATA","Soil carbon observations across Latin America","SOC stock at 0-30 cm. The completed extraction store contains 25,887 profiles and 174 predictor channels.")
map(.055,.16,.55,.80,meta)
txt(.64,.737,format(nrow(meta),big.mark=","),2.4,C["blue"],TRUE)
txt(.64,.685,"profiles in the patch store",.95,C["muted"])
txt(.64,.593,"174",2.1,C["olive"],TRUE); txt(.64,.546,"aligned predictor channels",.95,C["muted"])
txt(.64,.465,"3 / 9 / 15",1.6,bold=TRUE); txt(.64,.422,"cells per spatial window",.95,C["muted"])
txt(.64,.315,"Spatial coverage is uneven.",1.05,bold=TRUE)
txt(.64,.246,"Validation and applicability must\naccount for where observations exist.",.94,C["muted"])
txt(.04,.065,"Profile locations over the bio1 valid footprint. Lambert azimuthal equal-area (spherical), centred at 75 W / 15 S; equal x/y scale.",.77,C["muted"])
finish()

sm <- csv(file.path(trial,"smoke/patches/patch_meta.csv"))
methods<-c("spatial","knndm","random","holdout","region")
plans<-lapply(methods,function(d)readRDS(file.path(trial,"smoke/tuning",d,"fold_plan.rds")))
if(any(vapply(plans,function(p)p$n_rows!=nrow(sm),logical(1)))) stop("Fold plan / metadata mismatch")
testsets<-lapply(plans,function(p)sm$sample_id[p$folds[[1]]$test])
if(!all(vapply(testsets,function(ids)setequal(ids,testsets[[1]]),logical(1)))) stop("Test sets are not shared")
start("designs",1800,1800)
heading("05 / VALIDATION GEOMETRY","Same observations, different validation questions","Fold 1 of each design, from a test run on 1% of the profiles. The geometry only; no model results.")
names<-c("Spatial blocks","kNNDM","Random folds","Holdout","Ecoregions")
for(i in 1:5) {
  j<-(i-1)%%2; row<-(i-1)%/%2; x<-.045+j*.49; y<-.535-row*.235
  txt(x,y+.255,paste0(letters[i],"   ",names[i]),1.07,bold=TRUE)
  p<-plans[[i]]; f<-p$folds[[1]]; role<-rep("out",nrow(sm))
  role[f$train]<-"train"; role[f$validation]<-"validation"; role[f$test]<-"test"
  map(x+.04,y,x+.275,y+.235,sm,role)
  txt(x+.302,y+.159,paste0(length(f$train)," train"),.74,C["muted"])
  txt(x+.302,y+.125,paste0(length(f$validation)," validate"),.74,C["blue"])
  txt(x+.302,y+.091,paste0(length(f$test)," test"),.74,C["earth"])
}
legend(.55,.255,legend=c("Training","Validation","Shared test set","Not used in this fold"),
       col=c(C["train"],C["blue"],C["earth"],C["muted"]),pch=c(16,16,17,1),bty="n",cex=.9,y.intersp=1.7)
txt(.535,.095,paste0(nrow(sm)," profiles / fold 1\nOne test set for all five designs"),.79,C["muted"])
txt(.04,.03,"Spatial plan: 0.1-degree blocks; buffer 0.0337 degrees. Maps: spherical Lambert azimuthal equal-area, 75 W / 15 S; equal x/y scale.",.72,C["muted"])
finish()
writeLines(c("soilcnn vignette figures",paste("Source:",trial),paste("Full profiles:",nrow(meta)),
             paste("Example sample ID:",meta$sample_id[here]),"Validation panels: stored smoke plans, fold 1; shared test IDs verified.",
             "No trained predictions displayed. Raster ranges are shared across scales within each channel."),file.path(records,"figure_sources.txt"))
message("Editorial figures written to ",fig)

# Optional high-resolution input illustration.
if (file.exists(file.path(root,"tools/figure_inputs/28m/manifesto_recortes.csv"))) {
  source(file.path(root,"tools/vignette_highres.R"), local = TRUE)
}

# Explicitly simulated teaching example; independent of unfinished tuning.
source(file.path(root,"tools/vignette_selection.R"), local = TRUE)

# Full-plan detail, independent of trained checkpoints.
source(file.path(root,"tools/vignette_buffer.R"), local = TRUE)
