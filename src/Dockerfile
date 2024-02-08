FROM python:3.8.10-slim

WORKDIR /app

# install linux package dependencies


# can copy files only from current working directory where docker builds
# cannot copy files from arbitrary directories

COPY ./resnet_50_deeplab_v3+/oil_spill_seg_resnet_50_deeplab_v3+_80.pt /data/models/oil_spill_seg_resnet_50_deeplab_v3+_80.pt
COPY ./sample_padding_image_for_inference/img_0814.jpg /data/images/img_0814.jpg
COPY ./requirements_deployment.txt .

RUN pip install -r requirements_deployment.txt

COPY ./training/__init__.py ./training/
COPY ./training/dataset.py ./training/
COPY ./training/encoder_models.py ./training/
COPY ./training/decoder_models.py ./training/
COPY ./training/image_preprocessing.py ./training/
COPY ./training/logger_utils.py ./training/
COPY ./training/metrics.py ./training/
COPY ./training/seg_models.py ./training/
COPY ./training/image_stats.json ./training/
COPY ./app.py .

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000"]
