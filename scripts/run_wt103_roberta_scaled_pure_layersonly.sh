#!/bin/bash

source activate small-scale 

datapath=/data/sls/temp/belinkov/lm-generalization/data/wikitext-103/data-
# datapath=/scratch/belinkov/small-scale/data/wikitext-103/data-
databin=roberta-data-bin
exprdir=/data/sls/temp/belinkov/small-scale/expr/wikitext-103
# exprdir=/scratch/belinkov/small-scale/expr/wikitext-103


function train_scaled(){
    dsdiv=$1
    width=$2
    lr=$3
    gpus=$4
    # baseline:
    # encoder_layers=$(python -c "print(12 * 1)")
    # encoder_embed_dim=$(python -c "print(12 * 64)")
    # encoder_ffn_embed_dim=$(python -c "print(12 * 64 * 4)")
    # encoder_attention_heads=$(python -c "print(12 * 1)")

    encoder_layers=$(python -c "print($width)") #baseline is 12 . suggest to scan 1,2,4,8
    encoder_embed_dim=$(python -c "print(12 * 64)") 
    encoder_ffn_embed_dim=$(python -c "print(12 * 64 * 4)")
    encoder_attention_heads=$(python -c "print(12)") 

    dataset=wt103
    datadir=${datapath}$dsdiv/$databin
    ds_fraction=$(python -c "print(1/$dsdiv)")
    logpath=$exprdir/roberta.d$dsdiv.w$width.lr$lr.pure.gpus$gpus.log 
    checkpointsdir=$exprdir/roberta-checkpoints-d${dsdiv}-w${width}-lr${lr}-pure-gpus${gpus}

    echo 'Run training... '
    echo "dsdiv: $dsdiv"
    echo "width: $width"
    echo "lr: $lr"
    echo $dataset
    echo $datadir
    echo "log: $logpath"
    echo "encoder_layers: $encoder_layers"
    echo "encoder_embed_dim: $encoder_embed_dim"
    echo "encoder_ffn_embed_dim: $encoder_ffn_embed_dim"
    echo "encoder_attention_heads: $encoder_attention_heads"

    TOTAL_UPDATES=125000    # Total number of training steps
    #TOTAL_UPDATES=100    # Total number of training steps
    #WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
    WARMUP_UPDATES=0    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16         # Increase the batch size 16x


    # --arch roberta_1_64_256_1 \
    # fairseq-train --fp16 $datadir \
    fairseq-train $datadir \
        --task masked_lm \
        --criterion masked_lm \
        --arch roberta \
        --encoder-layers $encoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
        --encoder-attention-heads $encoder_attention_heads \
        --sample-break-mode complete \
        --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr $lr --warmup-updates $WARMUP_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
        --no-epoch-checkpoints --save-dir $checkpointsdir \
        --valid-subset 'valid,test' --skip-invalid-size-inputs-valid-test \
        --num-workers 4 --ddp-backend no_c10d --distributed-no-spawn --patience 10 &> $logpath 

}

gpus=8
#for w in 1 2 3 4 5	
#for w in 1 5  	
for w in 1 2 4 8
do
	#for d in 32 16 8 4
	#for d in 32 4    
	for d in 32 16 8 4
	do
		#for lr in 0.0005 0.00005 0.000005 
		for lr in 0.0005 
		do 
			train_scaled $d $w $lr $gpus
		done
	done
done




# for the record, if we want to save best/last checkpoints need to set this instead of --no-save
# --no-epoch-checkpoints --save-dir $checkpointsdir \
# for disabling validation, do:
# --disable-validation \
# for not saving checkpoints
# --no-save \
# validate/save only every 10 epochs
# --validate-interval 10 --save-interval 10 \

# for lr schedule 
# --lr-scheduler polynomial_decay \
# --lr $lr --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
	
