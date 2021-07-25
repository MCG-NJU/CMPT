mkdir -p logs
export CUDA_VISIBLE_DEVICES=$1
echo Using GPU Device "$1"
export PYTHONUNBUFFERED="True"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

starttime=$(date "+%Y-%m-%d-%H-%M-%S")
LOG="logs/traininginfo."$starttime
exec &> >(tee -a "$LOG")
echo Logging to "$LOG"
cat train.sh

for i in $(seq 0 1)
do
    python main.py $* starttime=$starttime seed=1
done

