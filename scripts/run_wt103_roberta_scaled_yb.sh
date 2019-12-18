#!/bin/bash

source activate small-scale 

datapath=/data/sls/temp/belinkov/lm-generalization/data/wikitext-103/data-
databin=roberta-data-bin
exprdir=/data/sls/temp/belinkov/small-scale/expr/wikitext-103


function train_scaled(){
    dsdiv=$1
    width=$2
    encoder_layers=$(python -c "print($width * 1)")
    encoder_embed_dim=$(python -c "print($width * 64)")
    encoder_ffn_embed_dim=$(python -c "print($encoder_embed_dim * 4)")
    encoder_attention_heads=$(python -c "print($width * 1)")

    dataset=wt103
    datadir=${datapath}$dsdiv/$databin
    ds_fraction=$(python -c "print(1/$dsdiv)")
    logpath=$exprdir/roberta.d$dsdiv.w$width.log 
    checkpointsdir=$exprdir/roberta-checkpoints-d${dsdiv}-w${width}

    echo 'Run training... '
    echo "dsdiv: $dsdiv"
    echo "width: $width"
    echo $dataset
    echo $datadir
    echo "log: $logpath"
    echo "encoder_layers: $encoder_layers"
    echo "encoder_embed_dim: $encoder_embed_dim"
    echo "encoder_ffn_embed_dim: $encoder_ffn_embed_dim"
    echo "encoder_attention_heads: $encoder_attention_heads"

    #TOTAL_UPDATES=125000    # Total number of training steps
    TOTAL_UPDATES=100    # Total number of training steps
    WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16         # Increase the 


    # --arch roberta_1_64_256_1 \
    fairseq-train --fp16 $datadir \
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
        --lr-scheduler polynomial_decay \
        --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --no-epoch-checkpoints --save-dir $checkpointsdir --valid-subset 'valid,test' --skip-invalid-size-inputs-valid-test &> $logpath 

}
for w in 1 
do
    train_scaled 32 $w
done

