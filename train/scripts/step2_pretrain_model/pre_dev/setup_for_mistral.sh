#!/bin/bash

set -e
echo ""

###copy common file to megatron-deepspeed
cp ../common/transformer.py ../../Megatron-DeepSpeed/megatron/model/transformer.py 
cp ../common/argument.py ../../Megatron-DeepSpeed/megatron/arguments.py
cp ../common/transformer_config.py ../../Megatron-DeepSpeed/megatron/core/transformer/transformer_config.py