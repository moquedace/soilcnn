# Actual full-run spatial plan: show one validation location and its buffer.
bp<-readRDS(file.path(trial,"full/tuning/spatial/fold_plan.rds"))
stopifnot(bp$n_rows==nrow(meta),bp$params$buffer_metric=="chebyshev")
bf<-bp$folds[[1]];br<-rep("out",nrow(meta))
br[bf$train]<-"train";br[bf$validation]<-"validation";br[bf$test]<-"test"
buffer<-bp$params$buffer
candidate<-which(br=="validation" & meta$x > -58 & meta$x < -40 & meta$y > -34 & meta$y < -14)
train_ids<-which(br=="train");out_ids<-which(br=="out")
score<-vapply(candidate,function(i) {
 nt<-min(pmax(abs(meta$x[train_ids]-meta$x[i]),abs(meta$y[train_ids]-meta$y[i])))
 no<-min(pmax(abs(meta$x[out_ids]-meta$x[i]),abs(meta$y[out_ids]-meta$y[i])))
 if(nt<.14 && no<buffer) nt else Inf
},numeric(1))
if(!any(is.finite(score)))stop("No suitable local validation/buffer example found")
bi<-candidate[which.min(score)];centre<-c(meta$x[bi],meta$y[bi])
dist_cheb<-pmax(abs(meta$x-centre[1]),abs(meta$y-centre[2]))
stopifnot(!any(br=="train" & dist_cheb<=buffer))
start("buffer",1800,1200)
heading("07 / SPATIAL BUFFER","Keep shared neighbourhoods out of training",
        "A real validation location from fold 1 of the full spatial plan. The shaded box shows its Chebyshev exclusion zone.")
local_panel<-function(x0,y0,x1,y1,half) {
 q<-which(dist_cheb<=half)
 metres_lon<-111320*cos(centre[2]*pi/180);metres_lat<-111132
 inches<-par("pin");sc<-min((x1-x0)*inches[1]/(2*half*metres_lon),(y1-y0)*inches[2]/(2*half*metres_lat))
 sx<-sc*metres_lon/inches[1];sy<-sc*metres_lat/inches[2]
 px<-function(v)(x0+x1)/2+(v-centre[1])*sx
 py<-function(v)(y0+y1)/2+(v-centre[2])*sy
 rect(px(centre[1]-half),py(centre[2]-half),px(centre[1]+half),py(centre[2]+half),col="#F6F8F7",border=C["line"])
 rect(px(centre[1]-buffer),py(centre[2]-buffer),px(centre[1]+buffer),py(centre[2]+buffer),col="#E7EFE1",border=C["olive"],lty=2,lwd=1.5)
 for(role in c("train","out","test","validation")) {
  ids<-q[br[q]==role];co<-c(train=unname(C["train"]),out=unname(C["muted"]),test=unname(C["earth"]),validation=unname(C["blue"]))[role]
  points(px(meta$x[ids]),py(meta$y[ids]),pch=if(role=="out")1 else if(role=="test")17 else 16,
         cex=if(role=="out").85 else .7,col=co,lwd=1)
 }
 points(px(centre[1]),py(centre[2]),pch=3,cex=1.2,lwd=1.6,col=C["ink"])
 # Geographic coordinates retained because buffering in this saved plan uses degrees.
 txt((x0+x1)/2,y0-.023,sprintf("%.4f W / %.4f S",abs(centre[1]),abs(centre[2])),.7,C["muted"],adj=.5)
}
txt(.045,.784,"a   Local context (1.2 degrees across)",1.05,bold=TRUE)
txt(.545,.784,"b   The exclusion zone",1.05,bold=TRUE)
# The context wide, the detail just around the zone: the two panels show two
# scales, not one twice.
half_context <- 0.6
half_detail  <- 1.6 * buffer
local_panel(.045,.28,.465,.75,half_context)
local_panel(.545,.28,.955,.75,half_detail)
# The legend lists only what the panels show.
shown <- unique(br[dist_cheb <= half_context])
keys  <- c(train="Training", validation="Validation", test="Test", out="Removed from training by a buffer")
marks <- c(train=16, validation=16, test=17, out=1)
cols  <- c(train=unname(C["train"]), validation=unname(C["blue"]), test=unname(C["earth"]), out=unname(C["muted"]))
show  <- intersect(names(keys), shown)
labels <- c(keys[show],"Selected validation location")
# Each entry as wide as its own label: the default gives every entry the
# longest one's width, spreading the short ones and crowding the last.
legend(.04,.193,legend=labels,text.width=strwidth(labels,cex=.79)+.03,
       col=c(cols[show],C["ink"]),pch=c(marks[show],3),horiz=TRUE,bty="n",cex=.79)
buffer_km <- buffer * 111.132
txt(.045,.102,sprintf("Buffer = %.4f degrees (about %.1f km, one 15 x 15 window), Chebyshev. No training observation falls inside the shaded zone.",buffer,buffer_km),.82,C["muted"])
txt(.045,.06,"Every validation and test location has such a zone; only one is drawn. Hollow points were removed from training by one of them.",.72,C["muted"])
finish()
writeLines(c("Real full-run spatial fold 1",paste("Selected sample ID:",meta$sample_id[bi]),
             paste("Centre:",paste(centre,collapse=", ")),paste("Buffer degrees:",buffer),
             "Verified: no training observation within the selected validation point's Chebyshev buffer."),
           file.path(records,"buffer_sources.txt"))
