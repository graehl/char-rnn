d=`dirname $0`
set -e
. $d/common.sh

data=data/$prefix
gold=$data/test.txt
testdata=$data/test.lower.txt

model=`ls $cv_dir/*.t7 | python $d/best_model.py`
beam=${beam:-8}
samplescript=sample.lua

output=$cv_dir/output.txt
echo "Truecasing using $model $size to $output"
main() {
    wc -l $testdata
    time th $d/$samplescript $model -beamsize $beam -verbose ${verbose:-1} -gpuid $gpu < $testdata > $output
    set -x
    time python $d/word_eval.py $gold $output
    head $output
}
main && exit 0 || exit 1
