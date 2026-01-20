python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode train \
  --method lowres --lowres-scale 0.5 \
  --output RN50_openai_train_lowres.pt

python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode test \
  --method lowres --lowres-scale 0.5 \
  --output RN50_openai_test_lowres.pt

python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode train \
  --method jpeg --jpeg-quality 50 \
  --output RN50_openai_train_jpeg50.pt

python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode test \
  --method jpeg --jpeg-quality 50 \
  --output RN50_openai_test_jpeg50.pt

python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode train \
  --method subject --subject-blur-radius 21 --subject-rect-scale 0.6 \
  --output RN50_openai_train_subject.pt

python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_degrade.py \
  --model-type RN50 --pretrained openai --mode test \
  --method subject --subject-blur-radius 21 --subject-rect-scale 0.6 \
  --output RN50_openai_test_subject.pt
