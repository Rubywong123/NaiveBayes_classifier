if [ -z $1 ]
then
    python3 main.py
else
    python3 main.py --dataset $1
fi