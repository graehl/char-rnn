d=`dirname $0`
set -e -x

. $d/common.sh

echo $model $size

main() {
    rm -f $cv_dir/*
    time th $d/train.lua \
         -min_freq ${minfreq:=50} \
         -maxvocab ${maxvocab:-255} \
         -data_dir $data \
         -model $model \
         -rnn_size $rnn \
         -num_layers $layer \
         -dropout $drop \
         -batch_size $batch_size \
         -seq_length $seq_len \
         -max_epochs $max_epochs \
         -learning_rate $learning_rate \
         -checkpoint_dir $cv_dir \
         -gpuid $gpu 2>&1 | tee $cv_dir/train.log
    cat $cv_dir/train.out
}
main && exit 0 || exit 1
