for SEED in 0 1 2 3 4
do
    export CUDA_VISIBLE_DEVICES=0 && python main.py --method aqmix+coach+vi --vi_lambda 0.001 --seed $SEED --centralized_every 2 > logs/sd$SEED-aqmix_coach_vi_T2.log 2>&1 &
    export CUDA_VISIBLE_DEVICES=1 && python main.py --method aqmix+coach+vi --vi_lambda 0.001 --seed $SEED --centralized_every 4 > logs/sd$SEED-aqmix_coach_vi_T4.log 2>&1 &
    export CUDA_VISIBLE_DEVICES=2 && python main.py --method aqmix+coach+vi --vi_lambda 0.001 --seed $SEED --centralized_every 8 > logs/sd$SEED-aqmix_coach_vi_T8.log 2>&1 &
    export CUDA_VISIBLE_DEVICES=3 && python main.py --method aqmix+coach                      --seed $SEED --centralized_every 4 > logs/sd$SEED-aqmix_coach_T4.log    2>&1 &
    export CUDA_VISIBLE_DEVICES=4 && python main.py --method aqmix+full                       --seed $SEED                       > logs/sd$SEED-aqmix+full.log        2>&1 &                       
    export CUDA_VISIBLE_DEVICES=5 && python main.py --method aqmix+period                     --seed $SEED                       > logs/sd$SEED-aqmix+period.log      2>&1 &
    export CUDA_VISIBLE_DEVICES=6 && python main.py --method aiqmix                           --seed $SEED                       > logs/sd$SEED-aiqmix.log            2>&1 &
    export CUDA_VISIBLE_DEVICES=7 && python main.py --method aqmix                            --seed $SEED                       > logs/sd$SEED-aqmix.log             2>&1 &
done
