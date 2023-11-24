SLURM_SCRIPT=slurm/cluster1_job_gpusrv5_a6000

rep=ilg
layers=4
aggr=mean

mkdir -p icaps24_slurm

for domain in blocksworld childsnack ferry floortile miconic rovers satellite sokoban spanner transport
do
    log_file=icaps24_slurm/cluster1_${rep}_${layers}_${aggr}_${domain}.log
    rm -f $log_file

    sbatch --job-name=${rep}_${domain} --output=$log_file $SLURM_SCRIPT "python3 scripts_gnn/train_test_ipc2023.py -r $rep -d $domain -a $aggr -l $layers"
done

    # "blocksworld",
    # "childsnack",
    # "ferry",
    # "floortile",
    # "miconic",
    # "rovers",
    # "satellite",
    # "sokoban",
    # "spanner",
    # "transport",
