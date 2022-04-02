if [ -z $1 ]
then
    python main.py
else
    python main.py --dataset $1
fi