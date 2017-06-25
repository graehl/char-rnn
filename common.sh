# wiki, wsj, conll_eng, conll_deu
prefix=$1

# lstm, gru
model=$2

# small, large
size=$3

# >= 0 for GPU, -1 for CPU
gpu=$4

if [ "$#" -ne 4 ] ; then
  echo "Usage: $0 <corpus> [lstm|gru] [small|large] <gpuid>" >&2
  exit 1
fi

# Hyperparameters
drop=0.2
batch_size=64
seq_len=96
data=data/$prefix
max_epochs=30
learning_rate=0.002
rnn=384
layer=2
if [ $size = "oldlarge" ]; then
  rnn=700
  layer=3
elif [[ $size = oldsmall ]] ; then
    rnn=300
    layer=2
elif [ $size = "large" ]; then
  rnn=768
  layer=3
fi

cv_dir=cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
mkdir -p cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
export LUA_PATH="$d/?.lua;$LUA_PATH"
if [[ $debug ]] ; then
    export CUDA_LAUNCH_BLOCKING=1
fi
