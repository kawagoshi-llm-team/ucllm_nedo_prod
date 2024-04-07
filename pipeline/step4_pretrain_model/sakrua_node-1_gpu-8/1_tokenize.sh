# tokenize

#data_path=$(yq -r '.data_path' config.yaml)
output_prefix=~/ucllm_nedo_prod/pipeline/step4_pretrain_model/input/tokenized_data
#$(yq -r '.output_prefix' config.yaml)
megatron_deepspeed_dir=~/ucllm_nedo_prod/pipeline/Megatron-DeepSpeed
#$(yq -r '.megatron_deepspeed_dir' config.yaml)
input_jsonl=~/ucllm_nedo_prod/pipeline/step00_download_datasets/output/refinedweb/refinedweb.jsonl
#$(yq -r '.input_jsonl' config.yaml)
input_tokenizer_file=~/ucllm_nedo_prod/pipeline/step3_tokenization/output/botchan.model
#$(yq -r '.input_tokenizer_file' config.yaml)
echo "tokenizer-model: ${input_tokenizer_file}"

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input  ${input_jsonl} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers 64 \
    --append-eod
echo ""