# Didactic values, deliberately separate from any trained model output.
# Base-graphics helpers are supplied by tools/vignette_figures.R.
demo <- data.frame(config_id=paste0("Example ",LETTERS[1:8]),
                   n_params=c(.25,.55,1,1.8,3,5,8,12)*1e6,
                   val_ccc_mean=c(.40,.455,.48,.492,.506,.515,.512,.510),
                   val_ccc_se=c(.024,.022,.020,.019,.018,.025,.020,.023))
# Mirror one_se(): best mean minus the best configuration's SE, then minimum
# complexity among eligible means. Error-bar overlap does not define eligibility.
best <- which.max(demo$val_ccc_mean)
threshold <- demo$val_ccc_mean[best]-demo$val_ccc_se[best]
eligible <- which(demo$val_ccc_mean>=threshold)
selected <- eligible[which.min(demo$n_params[eligible])]
stopifnot(best==6, selected==4, demo$val_ccc_mean[selected]>=threshold)
start("selection",1800,1150)
heading("06 / MODEL SELECTION","A simpler model can be the supported choice",
        "Didactic example / simulated values. Mean validation CCC with +/- 1 standard error; no trial performance is shown.")
x0<-.095;x1<-.70;y0<-.22;y1<-.735
xl<-c(.1,13);yl<-c(.36,.555)
xp<-function(x)x0+(x-xl[1])/diff(xl)*(x1-x0)
yp<-function(y)y0+(y-yl[1])/diff(yl)*(y1-y0)
rect(x0,yp(threshold),x1,y1,col="#EDF3E9",border=NA)
for(y in seq(.36,.54,.03)) {
 segments(x0,yp(y),x1,yp(y),col="#E1E7E5",lwd=.7)
 txt(x0-.012,yp(y),sprintf("%.2f",y),.8,C["muted"],adj=1)
}
for(x in c(0.25,1,3,5,8,12)) {
 segments(xp(x),y0,xp(x),y0-.008,col=C["muted"])
 txt(xp(x),y0-.029,format(x,trim=TRUE),.78,C["muted"],adj=.5)
}
segments(x0,y0,x1,y0,col=C["muted"],lwd=.8)
txt((x0+x1)/2,y0-.078,"Trainable parameters (millions)",.98,adj=.5)
text(.025,(y0+y1)/2,"Validation CCC",srt=90,cex=.98,col=C["ink"])
segments(x0,yp(threshold),x1,yp(threshold),col=C["olive"],lty=2,lwd=1.4)
for(i in seq_len(nrow(demo))) {
 x<-xp(demo$n_params[i]/1e6);y<-yp(demo$val_ccc_mean[i]);se<-demo$val_ccc_se[i]/diff(yl)*(y1-y0)
 color<-if(i==selected)C["earth"] else if(i==best)C["blue"] else C["muted"]
 segments(x,y-se,x,y+se,col=color,lwd=1.3)
 segments(x-.004,y-se,x+.004,y-se,col=color,lwd=1.3)
 segments(x-.004,y+se,x+.004,y+se,col=color,lwd=1.3)
 points(x,y,pch=if(i==selected)23 else 21,bg=if(i%in%c(best,selected))color else "white",col=color,cex=if(i%in%c(best,selected))1.3 else .95,lwd=1.5)
 txt(x,y-se-.019,LETTERS[i],.76,color,adj=.5)
}
txt(.755,.724,"Best mean / F",1.03,C["blue"],TRUE)
txt(.755,.671,"CCC 0.515\nSE 0.025",.88,C["muted"])
txt(.755,.584,"One-SE threshold",1.03,C["olive"],TRUE)
txt(.755,.538,"0.515 - 0.025 = 0.490",.81,C["muted"])
txt(.755,.443,"Selected / D",1.03,C["earth"],TRUE)
txt(.755,.375,"CCC 0.492 / 1.8M parameters\nSmallest eligible configuration",.82,C["muted"])
txt(.755,.276,"Eligibility uses the mean,\nnot error-bar overlap.",.85,C["muted"])
txt(.095,.062,"The shaded region marks eligible mean scores. The rule favours simplicity within one SE of the best; it is not a significance test.",.77,C["muted"])
finish()
write.csv(demo,file.path(records,"selection_example.csv"),row.names=FALSE)
