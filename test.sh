d=`dirname $0`
set -e -x

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

data=data/$prefix
testdata=$data/test.lower.txt
rnn=300
layer=2
if [ $size = "large" ]; then
  rnn=700
  layer=3
fi

cv_dir=cv/"$prefix"_"$model"_"$rnn"hidden_"$layer"layer
model=`ls $cv_dir/*.t7 | python $d/best_model.py`
beam=10
samplescript=sample.lua

export LUA_PATH="$d/?.lua;$LUA_PATH"
echo "Truecasing using $model $size"
time th $d/$samplescript $model -beamsize $beam -verbose 0 -gpuid $gpu < $testdata > $cv_dir/output.txt
