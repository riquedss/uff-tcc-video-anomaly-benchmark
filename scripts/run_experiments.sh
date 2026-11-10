#!/usr/bin/env bash
# Fila dos treinos finais: 3 modelos x 2 datasets x 3 sementes, todos com as
# features I3D 1024-d do PEL4VAD. Rodar na raiz do repo, no host (WSL2), de
# preferência dentro de um tmux.
#
#   scripts/run_experiments.sh            # roda o plano padrão
#   scripts/run_experiments.sh --smoke    # 1 treino curto de cada, CSV separado
#   PLAN="sultani sh 0" scripts/run_experiments.sh   # só as linhas indicadas
#
# Cada treino grava uma linha em results/green_metrics.csv e o log em
# results/logs/. Treinos concluídos ficam marcados em results/logs/*.done e
# são pulados se o script for executado de novo.
set -uo pipefail

cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

SH_I3D=${SH_I3D:-/data/shanghaitech/features/i3d}
UCF_I3D=${UCF_I3D:-/data/ucf-crime/features/i3d}
SEEDS=${SEEDS:-"0 1 2"}
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
LOGDIR=results/logs
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

default_plan() {
    for ds in sh ucf; do
        for model in sultani rtfm pel4vad; do
            for seed in $SEEDS; do echo "$model $ds $seed"; done
        done
    done
}

host_dir() {  # caminho no host de um caminho em /data
    echo "${DATA_ROOT:?defina DATA_ROOT no .env}${1#/data}"
}

warm_cache() {  # lê as features uma vez para o cache de disco (protocolo: cache quente)
    local dir size_kb avail_kb; dir=$(host_dir "$1")
    size_kb=$(du -sk "$dir" | cut -f1)
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$size_kb" -ge "$avail_kb" ]; then
        echo "[$(date +%T)] cache não aquecido: $dir ($((size_kb / 1048576)) GB) não cabe na RAM livre ($((avail_kb / 1048576)) GB)"
        return
    fi
    echo "[$(date +%T)] aquecendo cache: $dir"
    find "$dir" -name '*.npy' -print0 | xargs -0 cat > /dev/null
}

command_for() {  # model ds seed -> "servico|comando"
    local model=$1 ds=$2 seed=$3 feat log
    if [ "$ds" = sh ]; then feat=$SH_I3D; else feat=$UCF_I3D; fi
    log=/results/logs/${model}_${ds}_seed${seed}
    case $model in
        pel4vad)
            local epochs=""; [ $SMOKE = 1 ] && epochs="--max-epoch 1"
            echo "pel4vad|PEL4VAD_FEAT_PREFIX=$feat python main.py --dataset $ds --mode train --seed $seed $epochs --log-file $log.log"
            ;;
        rtfm)
            local name steps=""; [ "$ds" = sh ] && name=shanghai || name=ucf
            [ $SMOKE = 1 ] && steps="--max-epoch 20"
            echo "rtfm|python main.py --dataset $name --feature-size 1024 --feat-prefix $feat \
--rgb-list list/i3d/$ds-train.list --test-rgb-list list/i3d/$ds-test.list \
--gt /pel4vad_list/$ds/$ds-gt.npy --seed $seed --eval-every 0 --no-visdom $steps"
            ;;
        sultani)
            local label iters=""; [ "$ds" = sh ] && label=shanghaiTech || label=ucf-crime
            [ $SMOKE = 1 ] && iters="--epochs 1 --iterations_per_epoch 10"
            echo "anomaly-detection|python TrainingAnomalyDetector_public.py --features_path $feat \
--i3d_train_list list_i3d/$ds-train.list --i3d_test_list list_i3d/$ds-test.list \
--i3d_gt /pel4vad_list/$ds/$ds-gt.npy --dataset_name $label --seed $seed \
--exps_dir exps/i3d_${ds}_seed$seed --log_file $log.internal.log $iters"
            ;;
        *) echo "modelo desconhecido: $model" >&2; return 1 ;;
    esac
}

main() {
    mkdir -p "$LOGDIR"
    local csv_env=()
    if [ $SMOKE = 1 ]; then
        csv_env=(-e GREEN_METRICS_CSV=/results/smoke_metrics.csv)
        LOGDIR=results/logs/smoke; mkdir -p "$LOGDIR"
    fi

    echo "[$(date +%T)] construindo as imagens"
    $COMPOSE build anomaly-detection pel4vad rtfm || exit 1

    local plan; plan=$( [ -n "${PLAN:-}" ] && echo "$PLAN" || default_plan )
    [ $SMOKE = 1 ] && plan=$(printf "sultani sh 0\nrtfm sh 0\npel4vad sh 0\n")

    local warmed=""
    while read -r model ds seed; do
        [ -z "$model" ] && continue
        local name=${model}_${ds}_seed${seed}
        if [ -f "$LOGDIR/$name.done" ]; then
            echo "[$(date +%T)] já concluído: $name"; continue
        fi
        if [ $SMOKE = 0 ] && [[ " $warmed " != *" $ds "* ]]; then
            if [ "$ds" = sh ]; then warm_cache "$SH_I3D"; else warm_cache "$UCF_I3D"; fi
            warmed="$warmed $ds"
        fi
        local spec svc cmd
        spec=$(command_for "$model" "$ds" "$seed") || continue
        svc=${spec%%|*}; cmd=${spec#*|}
        [ $SMOKE = 1 ] && cmd=${cmd//\/results\/logs\//\/results\/logs\/smoke\/}
        echo "[$(date +%T)] início: $name"
        if $COMPOSE run --rm -T "${csv_env[@]}" "$svc" bash -c "$cmd" > "$LOGDIR/$name.out" 2>&1; then
            touch "$LOGDIR/$name.done"
            echo "[$(date +%T)] fim: $name"
        else
            echo "[$(date +%T)] FALHOU: $name (ver $LOGDIR/$name.out)"
        fi
    done <<< "$plan"
}

main "$@"
