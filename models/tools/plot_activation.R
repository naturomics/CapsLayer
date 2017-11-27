#! /bin/env Rscript

library(ggplot2)
library(animation)

path<-'results/'

plotPNG<-function(filename){
				fname<-unlist(strsplit(filename,split="[.]"))[1]
				step<-unlist(strsplit(fname,split="_"))[3]
				data<-data.frame(read.table(paste(path, filename, sep='')))
				data_dim<-dim(data)
				colnames(data)<-c(paste('activation',seq(1,data_dim[2]-1), sep=''), 'label')
				idx<-as.numeric(rownames(data))
				data<-data.frame(data)

				results<-data.frame()
				for(label in 0:(data_dim[2]-2)){
								subset<-data[data[,'label'] == label, ]
								results<-rbind(results,subset)
				}
				results<-cbind(results,idx)

				data<-data.frame()
				for(caps in 1:(data_dim[2]-1)){
								subset<-results[, c(caps,data_dim[2], data_dim[2]+1)]
								group<-rep(caps, data_dim[1])
								subset<-cbind(subset,group)
								colnames(subset)<-c('activation','label','sample_idx','group')
								data<-rbind(data,subset)
				}

				p<-ggplot(data, aes(x=sample_idx))
				p<-p + geom_line(aes(y=activation)) + facet_grid(group~.)
				main_title<-paste('The probability of entity presence(step', step, ')', sep='')
				p<-p + labs(title=main_title, y='activation probability', x='sample id') + theme(plot.title=element_text(hjust=0.5))
				ggsave(p, filename=paste(path, fname, '.png', sep=''))
}

filenames<-list.files(path, pattern="activations_step_.*txt")
for(file in filenames){plotPNG(file)}
