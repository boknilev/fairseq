#!/bin/bash

function train_scaled(){
    dsdiv=$1
    width=$2
    encoder_layers=$(python -c "print($width * 1)")
    encoder_embed_dim=$(python -c "print($width * 64)")
    encoder_ffn_embed_dim=$(python -c "print($encoder_embed_dim * 4)")
    encoder_attention_heads=$(python -c "print($width * 1)")

    dataset=wt103
    datadir=$(python -c "print('data-bin/wikitext-103-'+str($dsdiv)+'/')")
    ds_fraction=$(python -c "print(1/$dsdiv)")
    savepath=$(python -c "print('./transresults/width_'+str($width)+'_dsfraction_'+str($ds_fraction))")

    CUDA_VISIBLE_DEVICES=[6,7]
    echo 'Run training... '
    echo $dataset
    echo $datadir
    echo $savepath
    echo $(python -c "print('encoder_layers: '+str($encoder_layers))")
    echo $(python -c "print('encoder_embed_dim: '+str($encoder_embed_dim))")
    echo $(python -c "print('encoder_ffn_embed_dim: '+str($encoder_ffn_embed_dim))")
    echo $(python -c "print('encoder_attention_heads: '+str($encoder_attention_heads))")

    TOTAL_UPDATES=125000    # Total number of training steps
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
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1

}
for w in 1 
do
    train_scaled 32 $w
done

