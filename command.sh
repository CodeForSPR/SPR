#!/usr/bin/env bash

datafold="data"
logfold="loggings"

dataset="bitcoinAlpha"
#dataset="bitcoinOTC"
interval=10
epoches=2000

#dataset="epinions"
#dataset="slashdot"
#interval=200
#epoches=4000

if [ ! -d "${logfold}/${dataset}" ]; then
	mkdir -p "${logfold}/${dataset}"
fi

indices="0 1 2 3 4"
device="0"

for idx in ${indices}; do
	outputfile="${logfold}/${dataset}/${dataset}_${idx}"
	model_file="${outputfile}_model.pkl"
	log_file="${outputfile}_log.log"
	embed_file="${outputfile}_embedding.txt"

#	device=$(($idx % 4))
	python main.py \
		--cuda_device ${device} \
		--net_train ${datafold}/train_test/${dataset}/${dataset}_maxC_train${idx}.edgelist \
		--net_test ${datafold}/train_test/${dataset}/${dataset}_maxC_test${idx}.edgelist \
		--features_train ${datafold}/features/${dataset}/${dataset}_maxC_train${idx}_features64_tsvd.pkl \
		--model_path ${model_file} \
		--embedding_path ${embed_file} \
		--interval ${interval} \
		--epoches ${epoches} > ${log_file} 2>&1 &

done
