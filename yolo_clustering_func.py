{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOZope6QTV/HYX/WtoF16ED",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Daanish-Jain/drone-path/blob/main/yolo_clustering_func.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from ultralytics import YOLO\n",
        "import cv2\n",
        "import numpy as np\n",
        "import torch\n",
        "from sklearn.cluster import KMeans\n",
        "\n",
        "\n",
        "def load_backbone():\n",
        "\n",
        "    model = YOLO(\"yolov8l.pt\")\n",
        "    yolo_model = model.model\n",
        "\n",
        "    backbone = yolo_model.model[:10]\n",
        "    backbone.eval()\n",
        "\n",
        "    return backbone\n",
        "\n",
        "\n",
        "def preprocess_image(image):\n",
        "\n",
        "    image_resized = cv2.resize(image, (640,640))\n",
        "\n",
        "    x = torch.from_numpy(image_resized).permute(2,0,1).unsqueeze(0).float() / 255.0\n",
        "\n",
        "    return x, image_resized\n",
        "\n",
        "\n",
        "def extract_features(backbone, x):\n",
        "\n",
        "    with torch.no_grad():\n",
        "        features = backbone(x)\n",
        "\n",
        "    feat = features.squeeze(0).permute(1,2,0).contiguous().detach().cpu().numpy()\n",
        "\n",
        "    return feat\n",
        "\n",
        "\n",
        "def cluster_features(feature_map, k=4):\n",
        "\n",
        "    H, W, C = feature_map.shape\n",
        "\n",
        "    pixels = feature_map.reshape(-1, C)\n",
        "\n",
        "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
        "\n",
        "    labels = kmeans.fit_predict(pixels)\n",
        "\n",
        "    seg_map = labels.reshape(H, W)\n",
        "\n",
        "    return seg_map\n",
        "\n",
        "\n",
        "def find_road_cluster(seg_map):\n",
        "\n",
        "    H, W = seg_map.shape\n",
        "    clusters = np.unique(seg_map)\n",
        "\n",
        "    scores = []\n",
        "\n",
        "    for c in clusters:\n",
        "\n",
        "        mask = seg_map == c\n",
        "        lower_half = mask[H//2:, :]\n",
        "\n",
        "        scores.append(lower_half.sum())\n",
        "\n",
        "    road_cluster = clusters[np.argmax(scores)]\n",
        "\n",
        "    return road_cluster\n",
        "\n",
        "\n",
        "def visualize_clusters(image, seg_map, k):\n",
        "\n",
        "    # Resize cluster map to match image size\n",
        "    seg_resized = cv2.resize(\n",
        "        seg_map.astype(\"uint8\"),\n",
        "        (image.shape[1], image.shape[0]),\n",
        "        interpolation=cv2.INTER_NEAREST\n",
        "    )\n",
        "\n",
        "    # Generate consistent colors\n",
        "    np.random.seed(42)\n",
        "    colors = np.random.randint(0,255,(k,3))\n",
        "\n",
        "    # Convert cluster IDs → RGB colors\n",
        "    cluster_color_map = colors[seg_resized]\n",
        "\n",
        "    # Blend with original image\n",
        "    overlay = (0.6 * image + 0.4 * cluster_color_map).astype(\"uint8\")\n",
        "\n",
        "    return overlay\n",
        "\n",
        "\n",
        "def run_yolo_clustering_func(image, backbone=None, k=4):\n",
        "\n",
        "    if backbone is None:\n",
        "        backbone = load_backbone()\n",
        "\n",
        "    x, image_resized = preprocess_image(image)\n",
        "\n",
        "    features = extract_features(backbone, x)\n",
        "\n",
        "    seg_map = cluster_features(features, k)\n",
        "\n",
        "    road_cluster = find_road_cluster(seg_map)\n",
        "\n",
        "    road_mask = (seg_map == road_cluster)\n",
        "\n",
        "    road_mask_resized = cv2.resize(\n",
        "        road_mask.astype(\"uint8\"),\n",
        "        (image.shape[1], image.shape[0]),\n",
        "        interpolation=cv2.INTER_NEAREST\n",
        "    )\n",
        "\n",
        "    cluster_img = visualize_clusters(image, seg_map, k)\n",
        "\n",
        "    return seg_map, road_cluster, road_mask_resized, cluster_img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 383
        },
        "id": "PkY9ZxkboDSG",
        "outputId": "ab72adaf-bddd-4d05-af30-171d72b99645"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'ultralytics'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_1655/2254458322.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0multralytics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mYOLO\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mcv2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcluster\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mKMeans\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'ultralytics'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ]
    }
  ]
}