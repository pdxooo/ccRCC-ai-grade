args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
root <- normalizePath(args[1], winslash="/", mustWork=TRUE)
dir.create(file.path(root,"outputs"),showWarnings=FALSE)
out <- file.path(root,"outputs")
library(survival)
source_dir <- file.path(dirname(root),"gx1_posthoc_v1_20261025","outputs")
grades <- read.csv(file.path(source_dir,"grading_predictions.csv"),check.names=FALSE)
risks <- read.csv(file.path(source_dir,"recurrence_he_risk.csv"),check.names=FALSE)
fits <- list(); diagnostic <- list(); coefs <- list(); sensitivity <- list(); family <- list(); warn <- character()
add <- function(dat,cohort,encoder,label,rhs,core=FALSE) {
  dat <- dat[dat$in_recurrence_set==1,]
  dat$time <- dat$time_months_censored84;dat$event <- dat$recurrence
  dat$isup_cat <- factor(ifelse(dat$isup>=3,"3-4",as.character(dat$isup)),levels=c("1","2","3-4"))
  stopifnot(all(dat$time>0),!anyDuplicated(dat$patient_id))
  form <- as.formula(paste("Surv(time,event) ~",rhs))
  fit <- coxph(form,data=dat,ties="breslow",x=TRUE,y=TRUE,model=TRUE,control=coxph.control(eps=1e-10,iter.max=100))
  id <- paste(cohort,encoder,label,sep="__")
  fits[[id]] <<- list(fit=fit,data=dat,cohort=cohort,encoder=encoder,label=label,core=core,rhs=rhs)
  su <- summary(fit);ncoef <- nrow(su$coefficients)
  coefs[[id]] <<- data.frame(id=id,cohort=cohort,encoder=encoder,model=label,term=rownames(su$coefficients),n=fit$n,events=fit$nevent,
    beta=su$coefficients[,"coef"],se=su$coefficients[,"se(coef)"],hr=su$conf.int[,"exp(coef)"],lower95=su$conf.int[,"lower .95"],upper95=su$conf.int[,"upper .95"],p_raw=su$coefficients[,"Pr(>|z|)"],core=core)
  for (tr in c("km","rank")) {
    z <- cox.zph(fit,transform=tr,terms=TRUE,global=TRUE)
    diagnostic[[paste(id,tr)]] <<- data.frame(id=id,cohort=cohort,encoder=encoder,model=label,core=core,n=fit$n,events=fit$nevent,transform=tr,term=rownames(z$table),chisq=z$table[,"chisq"],df=z$table[,"df"],p_raw=z$table[,"p"])
  }
}
withCallingHandlers({
 for(coh in c("korea","scotland")) {
  for(enc in sort(unique(grades$encoder))) {
   d <- grades[grades$cohort==coh & grades$encoder==enc,]
   add(d,coh,enc,"AI_ISUP_numeric","isup + ai_grade",enc=="phikon-v2")
   add(d,coh,enc,"AI_Leibovich","leibovich_fuhrman + ai_grade",enc=="phikon-v2")
  }
  d <- grades[grades$cohort==coh & grades$encoder=="phikon-v2",]
  add(d,coh,"phikon-v2","AI_ISUP_categorical","isup_cat + ai_grade",TRUE)
  for(enc in c("phikon-v2","uni")) {
   d <- risks[risks$cohort==coh & risks$encoder==enc,]
   add(d,coh,enc,"HE_Leibovich","leibovich_fuhrman + he_risk_standardised",TRUE)
  }
 }
},warning=function(w){warn<<-c(warn,conditionMessage(w));invokeRestart("muffleWarning")})
coef_table <- do.call(rbind,coefs);diag_table <- do.call(rbind,diagnostic)
stopifnot(length(fits)==42,all(is.finite(diag_table$p_raw)))
write.csv(coef_table,file.path(out,"cox_coefficients.csv"),row.names=FALSE)
write.csv(diag_table,file.path(out,"ph_diagnostics.csv"),row.names=FALSE)
screen <- coef_table[coef_table$model=="AI_ISUP_numeric" & coef_table$term=="ai_grade",]
screen$p_holm <- ave(screen$p_raw,screen$cohort,FUN=function(p)p.adjust(p,method="holm"))
stopifnot(nrow(screen)==18)
write.csv(screen,file.path(out,"encoder_associations_holm.csv"),row.names=FALSE)
pdf(file.path(out,"core_schoenfeld_diagnostics.pdf"),width=10,height=7,onefile=TRUE)
for(id in names(fits)) {
  item <- fits[[id]];if(!item$core)next
  z <- cox.zph(item$fit,transform="km")
  par(mfrow=c(1,2),mar=c(4,4,4,1),oma=c(0,0,2,0))
  for(j in seq_len(ncol(z$y))) {
    plot(z,var=j,resid=TRUE,se=TRUE,xlab="Follow-up time (months)",ylab="Estimated time-varying coefficient")
    title(colnames(z$y)[j])
  }
  mtext(id,outer=TRUE,cex=.8)
}
dev.off()
# Conditional diagnostic sensitivity, retaining all primary estimates.
for(id in names(fits)) {
 item <- fits[[id]];if(!item$core)next
 z <- diag_table[diag_table$id==id & diag_table$transform=="km",]
 if(!any(z$p_raw<.05))next
 flagged <- z$term[z$term!="GLOBAL" & z$p_raw<.05]
 if(length(flagged)==0)flagged <- z$term[z$term!="GLOBAL"]
 # Expand categorical term to indicator columns for time interactions.
 dat <- item$data;dat$isup2 <- as.numeric(dat$isup_cat=="2");dat$isup34 <- as.numeric(dat$isup_cat=="3-4")
 flagvars <- unlist(lapply(flagged,function(v)if(v=="isup_cat")c("isup2","isup34")else v))
 rhs <- if(item$label=="AI_ISUP_categorical")"isup2 + isup34 + ai_grade" else item$rhs
 fm <- as.formula(paste("Surv(time,event) ~",rhs,"+",paste(paste0("tt(",flagvars,")"),collapse=" + ")))
 extended <- coxph(fm,data=dat,ties="breslow",tt=function(x,t,...)x*log(t/12),control=coxph.control(eps=1e-10,iter.max=100))
 su <- summary(extended)
 sensitivity[[id]] <- data.frame(id=id,term=rownames(su$coefficients),beta=su$coefficients[,"coef"],se=su$coefficients[,"se(coef)"],hr_at_12_months_or_interaction_multiplier=su$conf.int[,"exp(coef)"],p_raw=su$coefficients[,"Pr(>|z|)"],lrt_p=pchisq(2*(extended$loglik[2]-item$fit$loglik[2]),df=length(flagvars),lower.tail=FALSE))
}
if(length(sensitivity))write.csv(do.call(rbind,sensitivity),file.path(out,"time_interaction_sensitivity.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"sessionInfo.txt"))
writeLines(warn,file.path(out,"warnings.txt"))
capture.output({cat("42 models fitted; 10 core fits.\n");print(diag_table[diag_table$core & diag_table$transform=="km",c("cohort","encoder","model","term","p_raw")],row.names=FALSE);cat("\nHolm screen\n");print(screen[,c("cohort","encoder","hr","p_raw","p_holm")],row.names=FALSE);cat("\nTime sensitivities:",length(sensitivity),"\n")},file=file.path(out,"SUMMARY.txt"))
cat(readLines(file.path(out,"SUMMARY.txt")),sep="\n")
