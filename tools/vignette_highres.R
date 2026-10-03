# High-resolution input illustration; called by tools/vignette_figures.R.
# Requires its base-graphics helpers. Reads only the small, preserved input crops.
if (!exists("highres_dir")) highres_dir <- file.path(root,"tools/figure_inputs/28m")
hm <- read.csv2(file.path(highres_dir,"manifesto_recortes.csv"),stringsAsFactors=FALSE)
hn <- c("ndvi","elevation","clay")
hm <- hm[match(hn,hm$variavel),]
stopifnot(!anyNA(hm$variavel),length(unique(hm$longitude))==1,
          length(unique(hm$latitude))==1)
hd <- lapply(seq_along(hn),function(j) {
 f<-file.path(highres_dir,hm$arquivo[j])
 if(requireNamespace("terra",quietly=TRUE)) {
  r<-terra::rast(f); d<-as.data.frame(r,xy=TRUE,na.rm=FALSE)
  names(d)<-c("x","y","z"); return(d)
 }
 exe<-Sys.which("gdal_translate")
 if(!nzchar(exe)) exe<-"C:/OSGeo4W/bin/gdal_translate.exe"
 tmp<-tempfile(fileext=".xyz");on.exit(unlink(tmp))
 status<-system2(exe,c("-q","-of","XYZ",shQuote(f),shQuote(tmp)),stdout=FALSE,stderr=FALSE)
 if(status!=0)stop("Cannot read high-resolution crop: ",f)
 d<-read.table(tmp,col.names=c("x","y","z"));d$z[abs(d$z)>1e15]<-NA;d
})
for(j in 2:3) {
 if(!isTRUE(all.equal(hd[[1]][,1:2],hd[[j]][,1:2],tolerance=1e-10)))
   stop("The high-resolution input crops are not aligned")
}
hp<-c(hm$longitude[1],hm$latitude[1])
# Central cell is determined by nearest centre, avoiding floating boundary issues.
hcx<-sort(unique(hd[[1]]$x));hcy<-sort(unique(hd[[1]]$y),decreasing=TRUE)
hcol<-which.min(abs(hcx-hp[1]));hrow<-which.min(abs(hcy-hp[2]))
if(hcol<=7 || hrow<=7 || hcol+7>length(hcx) || hrow+7>length(hcy))
 stop("Crop is too small to contain the largest centred patch")
hlim<-lapply(hd,function(d)range(d$z,na.rm=TRUE))
hpatchlim<-lapply(hd,function(d) {
 d<-d[d$x %in% hcx[(hcol-7):(hcol+7)] & d$y %in% hcy[(hrow-7):(hrow+7)],]
 range(d$z,na.rm=TRUE)
})
if(any(vapply(hpatchlim,function(z)diff(z)<=0,logical(1))))
 stop("A channel is constant across the largest patch; use a different illustrative location")
if(any(vapply(hlim,function(z)!all(is.finite(z)) || diff(z)<=0,logical(1))))
 stop("A channel is empty or constant across the entire landscape crop")
# Approximate ground dimensions, calculated from the actual angular resolution.
metres_lat<-111132.92-559.82*cos(2*hp[2]*pi/180)+1.175*cos(4*hp[2]*pi/180)
metres_lon<-111412.84*cos(hp[2]*pi/180)-93.5*cos(3*hp[2]*pi/180)
pixel_m<-c(hm$resolucao_x[1]*metres_lon,hm$resolucao_y[1]*metres_lat)
start("patches",1800,2250)
heading("02 / MODEL INPUT","From landscape to multiscale patches",
        "An aligned high-resolution example at the same soil profile. Native raster cells are preserved; no interpolation is applied.")
htitles<-c("NDVI / unitless","Elevation / m","Clay / g/kg")
left<-c(.155,.445,.735); pw<-.215
for(j in 1:3) {
 x<-left[j]; txt(x,.808,htitles[j],1.02,bold=TRUE)
 tile(hd[[j]],x,.625,x+pw,.784,pal[[j]],hlim[[j]])
 for(k in 1:80) rect(x+(k-1)*pw/80,.61,x+k*pw/80,.616,col=pal[[j]][k],border=NA)
 txt(x,.599,format(round(hlim[[j]][1],2),trim=TRUE),.60,C["muted"])
 txt(x+pw,.599,format(round(hlim[[j]][2],2),trim=TRUE),.60,C["muted"],adj=1)
 # Nested native-cell windows, positioned relative to the entire raster crop.
 nx<-length(hcx);ny<-length(hcy)
 for(w in c(15,9,3)) {
  dx<-pw/nx;dy<-.159/ny
  xx<-x+(hcol-.5)*dx;yy<-.784-(hrow-.5)*dy
  rect(xx-w*dx/2,yy-w*dy/2,xx+w*dx/2,yy+w*dy/2,
       border="white",lwd=3)
  rect(xx-w*dx/2,yy-w*dy/2,xx+w*dx/2,yy+w*dy/2,
       border=C["ink"],lwd=1.2,lty=if(w==3)1 else if(w==9)2 else 3)
 }
 for(i in 1:3) {
  w<-c(3,9,15)[i];half<-(w-1)/2
  d<-hd[[j]];d<-d[d$x %in% hcx[(hcol-half):(hcol+half)] &
                  d$y %in% hcy[(hrow-half):(hrow+half)],]
  stopifnot(nrow(d)==w*w)
  y<-c(.411,.231,.051)[i]
  # Display each native cell as a square; panels share their display footprint.
  height<-pw*1800/2250
  tile(d,x,y,x+pw,y+height,pal[[j]],hpatchlim[[j]])
  if(i==1) {
   # Delicate boundaries make the 3 x 3 input explicit even on uniform channels.
   for(k in 1:2) {
    segments(x+k*pw/3,y,x+k*pw/3,y+height,col=adjustcolor("white",.45),lwd=.7)
    segments(x,y+k*height/3,x+pw,y+k*height/3,col=adjustcolor("white",.45),lwd=.7)
   }
  }
  points(x+pw/2,y+height/2,pch=21,bg="white",col=C["ink"],cex=.65,lwd=1)
 }
 for(k in 1:80) rect(x+(k-1)*pw/80,.032,x+k*pw/80,.04,col=pal[[j]][k],border=NA)
 txt(x,.022,format(round(hpatchlim[[j]][1],2),trim=TRUE),.64,C["muted"])
 txt(x+pw,.022,format(round(hpatchlim[[j]][2],2),trim=TRUE),.64,C["muted"],adj=1)
}
txt(.035,.693,"Context",.88,bold=TRUE)
txt(.035,.666,"~4 km",.74,C["muted"])
for(i in 1:3) {
 w<-c(3,9,15)[i];y<-c(.411,.231,.051)[i]+.086
 txt(.035,y+.012,paste0(w," x ",w),.95,bold=TRUE)
 txt(.035,y-.017,sprintf("~%d x %d m",round(w*pixel_m[1]),round(w*pixel_m[2])),.69,C["muted"])
}
txt(.155,.007,sprintf("Cells: ~%.1f x %.1f m. Separate landscape and patch colour ranges; the three window sizes share one range per channel.",pixel_m[1],pixel_m[2]),.65,C["muted"])
finish()
writeLines(c(paste("High-resolution source crops:",highres_dir),
             paste("Central profile:",hp[1],hp[2]),
             paste("Angular resolution:",hm$resolucao_x[1],hm$resolucao_y[1]),
             paste("Approximate cell dimensions in metres:",paste(round(pixel_m,3),collapse=" x ")),
             "Native input rasters are aligned; patches contain exactly 9, 81 and 225 cells.",
             "This is a separate input illustration, not the 250 m trial's training data."),
             file.path(records,"highres_sources.txt"))
message("High-resolution illustration written; approximate cell size: ",paste(round(pixel_m,1),collapse=" x ")," m")
