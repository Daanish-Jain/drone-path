{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOkUkibogFbT9M1hPae5Slk",
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
        "%%writefile yolo_clustering_func.py\n",
        "from ultralytics import YOLO\n",
        "import cv2\n",
        "import numpy as np\n",
        "import torch\n",
        "from sklearn.cluster import KMeans\n",
        "\n",
        "\n",
        "_backbone = None\n",
        "\n",
        "def load_backbone():\n",
        "\n",
        "    global _backbone\n",
        "\n",
        "    if _backbone is None:\n",
        "        model = YOLO(\"yolov8l.pt\")\n",
        "        yolo_model = model.model\n",
        "        _backbone = yolo_model.model[:10]\n",
        "        _backbone.eval()\n",
        "\n",
        "    return _backbone\n",
        "\n",
        "\n",
        "def preprocess_image(image):\n",
        "\n",
        "    image_resized = cv2.resize(image, (640,640))\n",
        "\n",
        "    x = torch.from_numpy(image_resized).permute(2,0,1).unsqueeze(0).float() / 255.0\n",
        "\n",
        "    return x\n",
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
        "    alpha = 0.4\n",
        "    overlay = ((1-alpha) * image + alpha * cluster_color_map).astype(\"uint8\")\n",
        "\n",
        "    return overlay\n",
        "\n",
        "\n",
        "def run_yolo_clustering_func(image, backbone=None, k=4):\n",
        "\n",
        "    if backbone is None:\n",
        "        backbone = load_backbone()\n",
        "\n",
        "    x = preprocess_image(image)\n",
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
          "base_uri": "https://localhost:8080/"
        },
        "id": "PkY9ZxkboDSG",
        "outputId": "4ff43223-ce22-47f4-a220-921c501f0a4a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting yolo_clustering_func.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/Daanish-Jain/drone-path.git"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GPLOm8lOiXSb",
        "outputId": "baacbe34-c909-4cb6-a520-06b549345d70"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "fatal: destination path 'drone-path' already exists and is not an empty directory.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /content/drone-path"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R7ZdQrJqjRCQ",
        "outputId": "ab1feafe-3cdf-49f6-93d2-22e90de11e12"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "drone-path  README.md  yolo_clustering.ipynb\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!mv yolo_clustering_func.py drone-path/"
      ],
      "metadata": {
        "id": "dbQq5tNZiKcr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!ls /content/drone-path"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nVwazn4tjq21",
        "outputId": "8438abec-a0c0-42c6-b1b0-4d831e977209"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "drone-path  README.md  yolo_clustering.ipynb\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%cd drone-path"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "73cJtpT6iLCe",
        "outputId": "68d6734c-31df-4cb6-9e8b-31aaa2b452c1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/drone-path/drone-path\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git add yolo_clustering_func.py\n",
        "!git commit -m \"Updated clustering module\"\n",
        "!git push"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D3cwxf2LiPCN",
        "outputId": "b8372478-151e-4069-ea52-c0e68fa2653c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Author identity unknown\n",
            "\n",
            "*** Please tell me who you are.\n",
            "\n",
            "Run\n",
            "\n",
            "  git config --global user.email \"you@example.com\"\n",
            "  git config --global user.name \"Your Name\"\n",
            "\n",
            "to set your account's default identity.\n",
            "Omit --global to set the identity only in this repository.\n",
            "\n",
            "fatal: unable to auto-detect email address (got 'root@f18011320b7c.(none)')\n",
            "fatal: could not read Username for 'https://github.com': No such device or address\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Cq8isnQ3iRRE"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}