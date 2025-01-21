# pull docker
echo "for details and usage examples, see https://github.com/mit-han-lab/qserve/tree/main?tab=readme-ov-file#usage-and-examples"
docker pull nyunadmin/qserve:latest

# get this dir from file
this_dir=$(dirname $(readlink -f $0))


PRECISION=w4a8kv4 # change accordingly
GROUP_SIZE=128 # change accordingly
MODEL_PATH=$(basename "$this_dir")

# echo all
echo "this_dir: $this_dir"
echo "PRECISION: $PRECISION"
echo "GROUP_SIZE: $GROUP_SIZE"
echo "MODEL_PATH: $MODEL_PATH"


# run docker
docker run \
    --gpus all -it \
    -v $this_dir:/llm/$MODEL_PATH \
    --workdir /llm/$MODEL_PATH \
    --rm \
    nyunadmin/qserve:latest \
    python qserve_e2e_generation.py \
    --model /llm/$MODEL_PATH \
    --ifb-mode \
    --precision $PRECISION \
    --quant-path /llm/$MODEL_PATH \
    --group-size $GROUP_SIZE
