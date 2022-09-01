#please download the data set and locate it under ./datasets
set -e
dir=$(dirname $(readlink -fn --  $0))
cd ../
PYTHONPATH=./Step1_Detection python -m Step1_Detection.utils.test

cd $dir
PYTHONPATH=./ python ./identifier/test_from_pkl.py --root ./exp

python Step3_Matching/nmf.py