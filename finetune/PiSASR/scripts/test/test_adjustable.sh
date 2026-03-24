
python test_pisasr.py \
--pretrained_model_path preset/models/stable-diffusion-2-1-base \
--pretrained_path preset/models/pisa_sr.pkl \
--process_size 512 \
--upscale 4 \
--input_image preset/test_datasets \
--output_dir experiments/test \
--lambda_pix 1.0 \
--lambda_sem 1.0
