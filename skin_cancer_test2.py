import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50, ResNet50_Weights
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar
import warnings
import random
from typing import List, Tuple, Dict, Optional
import time
import pickle
import xgboost as xgb
import os

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#####################
# 1. DATA PREPARATION
#####################

# Define transforms with enhanced augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset paths
base_data_path = Path(r"D:\college - Sri eshwar\sumathi mam\datasets\skin_cancer1\Skin cancer ISIC The International Skin Imaging Collaboration")
train_data_path = base_data_path / 'Train'
test_data_path = base_data_path / 'Test'

# Checkpoint directory
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)
binary_checkpoint_path = checkpoint_dir / 'binary_model.pkl'
cnn_checkpoint_path = checkpoint_dir / 'cnn_model.pth'
benign_checkpoint_path = checkpoint_dir / 'benign_model.pth'
malignant_checkpoint_path = checkpoint_dir / 'malignant_model.pth'

# Load datasets
try:
    binary_train_dataset = ImageFolder(train_data_path, transform=transform)
    binary_test_dataset = ImageFolder(test_data_path, transform=transform)
    print("Binary Classes:", binary_train_dataset.classes)  # ['benign', 'malignant']

    benign_train_dataset = ImageFolder(train_data_path / 'benign', transform=transform)
    malignant_train_dataset = ImageFolder(train_data_path / 'malignant', transform=transform)
    benign_test_dataset = ImageFolder(test_data_path / 'benign', transform=transform)
    malignant_test_dataset = ImageFolder(test_data_path / 'malignant', transform=transform)
    print("Benign Subcategories:", benign_train_dataset.classes)
    print("Malignant Subcategories:", malignant_train_dataset.classes)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Define subcategory class names
benign_classes = ['Dermatofibroma', 'Vascular_Lesion', 'Pigmented_Benign_Keratosis', 'Nevus', 'Seborrheic_Keratosis']
malignant_classes = ['Squamous_Cell_Carcinoma', 'Melanoma', 'Basal_Cell_Carcinoma', 'Actinic_Keratosis']

# Split binary dataset
total_len = len(binary_train_dataset)
train_len = int(0.8 * total_len)
val_len = total_len - train_len
binary_train_subset, binary_val_subset = torch.utils.data.random_split(binary_train_dataset, [train_len, val_len])

# DataLoaders
train_dataloader = DataLoader(binary_train_subset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(binary_val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(binary_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
benign_train_dataloader = DataLoader(benign_train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
malignant_train_dataloader = DataLoader(malignant_train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
benign_test_dataloader = DataLoader(benign_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
malignant_test_dataloader = DataLoader(malignant_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

#########################
# 2. FEATURE EXTRACTION
#########################

class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        def hook(module, input, output):
            self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, x):
        self.features = []
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            _ = self.model(x)
        avg_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        fmap_size = self.features[0].shape[-2]
        adaptive_pool = nn.AdaptiveAvgPool2d((fmap_size, fmap_size))
        resized_maps = [adaptive_pool(avg_pool(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, dim=1)
        return patch

class LightweightCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EnsemblePSO:
    def __init__(self, n_particles=50, max_iter=200):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = 0.7
        self.c1 = 2.0
        self.c2 = 2.0

    def _initialize_particles(self, n_models):
        particles = np.random.rand(self.n_particles, n_models)
        particles = particles / particles.sum(axis=1)[:, np.newaxis]
        velocities = np.random.randn(self.n_particles, n_models) * 0.1
        return particles, velocities

    def _evaluate_fitness(self, weights, predictions, true_labels):
        ensemble_pred = np.zeros_like(true_labels, dtype=float)
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        return accuracy_score(true_labels, ensemble_pred)

    def optimize(self, model_predictions, true_labels):
        n_models = len(model_predictions)
        particles, velocities = self._initialize_particles(n_models)
        pbest = particles.copy()
        pbest_fitness = np.array([self._evaluate_fitness(p, model_predictions, true_labels) for p in particles])
        gbest = pbest[pbest_fitness.argmax()]
        gbest_fitness = pbest_fitness.max()
        for iteration in range(self.max_iter):
            r1, r2 = np.random.rand(2, self.n_particles, n_models)
            velocities = (self.w * velocities + self.c1 * r1 * (pbest - particles) + self.c2 * r2 * (gbest - particles))
            particles += velocities
            particles = np.maximum(particles, 0)
            particles = particles / particles.sum(axis=1)[:, np.newaxis]
            fitness = np.array([self._evaluate_fitness(p, model_predictions, true_labels) for p in particles])
            improve_idx = fitness > pbest_fitness
            pbest[improve_idx] = particles[improve_idx]
            pbest_fitness[improve_idx] = fitness[improve_idx]
            if fitness.max() > gbest_fitness:
                gbest = particles[fitness.argmax()]
                gbest_fitness = fitness.max()
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {gbest_fitness:.4f}")
        return gbest, gbest_fitness

class HybridEnsemble:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.svm = SVC(kernel='rbf', probability=True, random_state=42, C=2.0)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        self.cnn = LightweightCNN(input_channels=input_shape[0], num_classes=num_classes)
        self.scaler = StandardScaler()
        self.ensemble_weights = None
        self.cnn.to(self.device)
        self.svm_train_time = 0
        self.rf_train_time = 0
        self.cnn_train_time = 0

    def _train_cnn(self, train_loader, val_loader, epochs=150):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        best_val_acc = 0
        best_weights = None
        for epoch in range(epochs):
            self.cnn.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            self.cnn.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.cnn(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = val_correct / val_total
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.cnn.state_dict().copy()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Loss = {running_loss/len(train_loader):.4f}, Validation Accuracy = {val_acc:.4f}")
        self.cnn.load_state_dict(best_weights)
        torch.cuda.empty_cache()  # Clear GPU memory
        return best_val_acc

    def fit(self, X, y, X_val=None, y_val=None, cnn_loader=None, val_loader=None, checkpoint_path=None):
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training SVM...")
        start_time = time.time()
        X_train_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_train_scaled, y)
        self.svm_train_time = time.time() - start_time
        
        print("Training Random Forest...")
        start_time = time.time()
        self.rf.fit(X_train_scaled, y)
        self.rf_train_time = time.time() - start_time
        
        print("Training CNN...")
        start_time = time.time()
        cnn_acc = self._train_cnn(cnn_loader, val_loader)
        self.cnn_train_time = time.time() - start_time
        
        print("Getting ensemble predictions...")
        X_val_scaled = self.scaler.transform(X_val)
        svm_pred = self.svm.predict_proba(X_val_scaled)[:, 1]
        rf_pred = self.rf.predict_proba(X_val_scaled)[:, 1]
        self.cnn.eval()
        cnn_pred = []
        y_val_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = torch.softmax(self.cnn(inputs), dim=1)
                cnn_pred.extend(outputs[:, 1].cpu().numpy())
                y_val_true.extend(labels.cpu().numpy())  # Ensure labels are on CPU for numpy
        cnn_pred = np.array(cnn_pred)
        y_val_true = np.array(y_val_true)
        
        print("Optimizing ensemble weights...")
        pso = EnsemblePSO()
        self.ensemble_weights, ensemble_fitness = pso.optimize([svm_pred, rf_pred, cnn_pred], y_val_true)
        print(f"\nFinal ensemble weights:\nSVM: {self.ensemble_weights[0]:.4f}\nRandom Forest: {self.ensemble_weights[1]:.4f}\nCNN: {self.ensemble_weights[2]:.4f}")
        print(f"Ensemble Validation Accuracy: {ensemble_fitness:.4f}")

        # Save checkpoint only after training
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)

    def predict(self, X, loader=None):
        X_scaled = self.scaler.transform(X)
        svm_pred = self.svm.predict_proba(X_scaled)[:, 1]
        rf_pred = self.rf.predict_proba(X_scaled)[:, 1]
        self.cnn.eval()
        cnn_pred = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = torch.softmax(self.cnn(inputs), dim=1)
                cnn_pred.extend(outputs[:, 1].cpu().numpy())
        cnn_pred = np.array(cnn_pred)
        ensemble_prob = self.ensemble_weights[0] * svm_pred + self.ensemble_weights[1] * rf_pred + self.ensemble_weights[2] * cnn_pred
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        torch.cuda.empty_cache()  # Clear GPU memory
        return ensemble_pred, ensemble_prob

    def _save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'svm': self.svm,
            'rf': self.rf,
            'scaler': self.scaler,
            'ensemble_weights': self.ensemble_weights,
            'svm_train_time': self.svm_train_time,
            'rf_train_time': self.rf_train_time,
            'cnn_train_time': self.cnn_train_time
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        torch.save(self.cnn.state_dict(), cnn_checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} and {cnn_checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.svm = checkpoint['svm']
        self.rf = checkpoint['rf']
        self.scaler = checkpoint['scaler']
        self.ensemble_weights = checkpoint['ensemble_weights']
        self.svm_train_time = checkpoint['svm_train_time']
        self.rf_train_time = checkpoint['rf_train_time']
        self.cnn_train_time = checkpoint['cnn_train_time']
        self.cnn = LightweightCNN(input_channels=self.input_shape[0], num_classes=self.num_classes)
        self.cnn.load_state_dict(torch.load(cnn_checkpoint_path))
        self.cnn.to(self.device)
        print(f"Checkpoint loaded from {checkpoint_path} and {cnn_checkpoint_path}")

###############################
# 3. GLCM EXTRACTION & VISUALIZATION
###############################

class GLCMExtractor:
    def __init__(self, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8):
        self.distances = distances
        self.angles = angles
        self.levels = levels

    def normalize_feature_map(self, feature_map):
        feature_map = np.array(feature_map, dtype=np.float32)
        min_val = feature_map.min()
        max_val = feature_map.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(feature_map, dtype=np.uint8)
        bins = np.linspace(min_val, max_val + 1e-10, self.levels + 1)
        digitized = np.digitize(feature_map, bins) - 1
        digitized = np.clip(digitized, 0, self.levels - 1)
        return digitized.astype(np.uint8)

    def calculate_glcm_features(self, feature_map):
        norm_feature_map = self.normalize_feature_map(feature_map)
        glcm = graycomatrix(
            norm_feature_map,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'ASM': graycoprops(glcm, 'ASM').mean()
        }
        return features, glcm

    def visualize_glcm(self, feature_map, glcm):
        num_distances = len(self.distances)
        num_angles = len(self.angles)
        fig = plt.figure(figsize=(15, 5 + 3 * num_distances))
        gs = plt.GridSpec(2 + num_distances, num_angles)
        ax_orig = fig.add_subplot(gs[0, :])
        im_orig = ax_orig.imshow(feature_map, cmap='gray')
        ax_orig.set_title('Original Feature Map')
        plt.colorbar(im_orig, ax=ax_orig)
        ax_norm = fig.add_subplot(gs[1, :])
        norm_map = self.normalize_feature_map(feature_map)
        im_norm = ax_norm.imshow(norm_map, cmap='gray')
        ax_norm.set_title(f'Quantized Feature Map ({self.levels} levels)')
        plt.colorbar(im_norm, ax=ax_norm)
        vmax = glcm.max()
        for i, d in enumerate(self.distances):
            for j, angle in enumerate(self.angles):
                ax = fig.add_subplot(gs[i+2, j])
                im = ax.imshow(glcm[:, :, i, j], cmap='viridis', interpolation='nearest', vmax=vmax)
                ax.set_title(f'Distance: {d}, Angle: {np.round(np.degrees(angle))}°')
                plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

def analyze_feature_map_texture(feature_map, glcm_extractor):
    features, glcm = glcm_extractor.calculate_glcm_features(feature_map)
    fig = glcm_extractor.visualize_glcm(feature_map, glcm)
    return features, fig

##########################################
# 4. HANDCRAFTED FEATURES (COLOR & SHAPE)
##########################################

class HandcraftedFeatureExtractor:
    def __init__(self, color_bins=32, fourier_points=64):
        self.color_bins = color_bins
        self.fourier_points = fourier_points

    def extract_color_features(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        features = {}
        color_spaces = {'rgb': image, 'hsv': hsv, 'lab': lab}
        for space_name, space_image in color_spaces.items():
            for i, channel_name in enumerate(['1', '2', '3']):
                hist, _ = np.histogram(space_image[:,:,i].ravel(), bins=self.color_bins, range=(0, 256), density=True)
                features[f'{space_name}_{channel_name}_hist'] = hist
                channel = space_image[:,:,i]
                features[f'{space_name}_{channel_name}_mean'] = np.mean(channel)
                features[f'{space_name}_{channel_name}_std'] = np.std(channel)
                features[f'{space_name}_{channel_name}_skew'] = np.mean(((channel - channel.mean())/channel.std())**3)
                features[f'{space_name}_{channel_name}_kurtosis'] = np.mean(((channel - channel.mean())/channel.std())**4)
        return features

    def extract_shape_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features = {}
        if not contours:
            features['area'] = 0
            features['perimeter'] = 0
            features['circularity'] = 0
            features['fourier_descriptors'] = np.zeros(self.fourier_points * 2)
            features['aspect_ratio'] = 0
            features['extent'] = 0
            features['solidity'] = 0
            return features
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        features['area'] = area
        features['perimeter'] = perimeter
        features['circularity'] = circularity
        x, y, w, h = cv2.boundingRect(max_contour)
        features['aspect_ratio'] = float(w)/h if h > 0 else 0
        features['extent'] = float(area)/(w*h) if w*h > 0 else 0
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = float(area)/hull_area if hull_area > 0 else 0
        resampled_contour = self._resample_contour(max_contour, self.fourier_points)
        fourier_descriptors = self._extract_fourier_descriptor(resampled_contour)
        features['fourier_descriptors'] = fourier_descriptors
        return features

    def _resample_contour(self, contour, num_points):
        contour = contour[:, 0, :]
        distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
        cum_distances = np.concatenate(([0], np.cumsum(distances)))
        total_length = cum_distances[-1]
        if total_length == 0:
            return np.zeros((num_points, 2))
        points_position = np.linspace(0, total_length, num_points)
        resampled_contour = np.zeros((num_points, 2))
        for i, pos in enumerate(points_position):
            idx = np.searchsorted(cum_distances, pos) - 1
            idx = max(0, min(idx, len(contour) - 2))
            alpha = (pos - cum_distances[idx]) / (cum_distances[idx + 1] - cum_distances[idx])
            resampled_contour[i] = contour[idx] + alpha * (contour[idx + 1] - contour[idx])
        return resampled_contour

    def _extract_fourier_descriptor(self, contour_points):
        centered = contour_points - np.mean(contour_points, axis=0)
        complex_coords = centered[:, 0] + 1j * centered[:, 1]
        fourier_descriptors = np.fft.fft(complex_coords)
        fourier_descriptors = np.abs(fourier_descriptors)
        if fourier_descriptors[1] != 0:
            fourier_descriptors = fourier_descriptors / fourier_descriptors[1]
        return np.concatenate([fourier_descriptors[:self.fourier_points].real, fourier_descriptors[:self.fourier_points].imag])

    def extract_features(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        else:
            image_pil = image_tensor
        image_rgb = image_pil.convert('RGB')
        image_np = np.array(image_rgb)
        color_features = self.extract_color_features(image_np)
        shape_features = self.extract_shape_features(image_np)
        features = {**color_features, **shape_features}
        return features

    def visualize_features(self, image_pil, features):
        image_np = np.array(image_pil.convert('RGB'))
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_np)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2 = fig.add_subplot(gs[0, 1])
        for space in ['rgb', 'hsv', 'lab']:
            for i, channel in enumerate(['1', '2', '3']):
                hist_key = f'{space}_{channel}_hist'
                if hist_key in features:
                    ax2.plot(features[hist_key], label=f'{space.upper()}_{channel}', alpha=0.7)
        ax2.set_title('Color Histograms')
        ax2.legend()
        ax3 = fig.add_subplot(gs[0, 2])
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ax3.imshow(thresh, cmap='gray')
        ax3.set_title('Shape Segmentation')
        ax3.axis('off')
        ax4 = fig.add_subplot(gs[1, :])
        shape_metrics = ['area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity']
        metrics_values = [features.get(metric, 0) for metric in shape_metrics]
        ax4.bar(shape_metrics, metrics_values)
        ax4.set_title('Shape Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

class BinaryPSO:
    def __init__(self, n_particles: int, n_features: int, max_iter: int = 100, w_max: float = 0.9, w_min: float = 0.4,
                 min_features: int = 10, max_features: int = 200, target_features: int = 100, mutation_rate: float = 0.01):
        self.n_particles = n_particles
        self.n_features = n_features
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.min_features = min_features
        self.max_features = max_features
        self.target_features = target_features
        self.mutation_rate = mutation_rate
        self.c1_start = 2.5
        self.c1_end = 0.5
        self.c2_start = 0.5
        self.c2_end = 2.5
        self.particles = self._initialize_diverse_swarm()
        self.velocities = np.random.randn(n_particles, n_features) * 0.1
        self.pbest = self.particles.copy()
        self.pbest_fitness = np.full(n_particles, float('-inf'))
        self.pbest_diversity = np.zeros(n_particles)
        self.gbest = np.zeros(n_features, dtype=bool)
        self.gbest_fitness = float('-inf')
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.best_fitness_streak = float('-inf')
        self.velocity_reset_counter = 0
        self._fitness_cache = {}

    def _initialize_diverse_swarm(self) -> np.ndarray:
        particles = []
        feature_proportions = np.linspace(0.1, 0.5, self.n_particles)
        for prop in feature_proportions:
            particle = np.random.rand(self.n_features) < prop
            while np.sum(particle) < self.min_features:
                zero_indices = np.where(particle == 0)[0]
                idx = np.random.choice(zero_indices)
                particle[idx] = True
            particles.append(particle)
        return np.array(particles)

    def _calculate_swarm_diversity(self) -> float:
        mean_particle = np.mean(self.particles, axis=0)
        distances = np.sum(np.abs(self.particles - mean_particle), axis=1)
        return np.mean(distances) / self.n_features

    def _adaptive_mutation(self, particle: np.ndarray) -> np.ndarray:
        diversity = self._calculate_swarm_diversity()
        adaptive_rate = self.mutation_rate * (1 + (1 - diversity))
        mutation_mask = np.random.rand(self.n_features) < adaptive_rate
        particle = particle.copy()
        particle[mutation_mask] = ~particle[mutation_mask]
        n_selected = np.sum(particle)
        if n_selected < self.min_features:
            zero_indices = np.where(particle == 0)[0]
            to_select = np.random.choice(zero_indices, size=self.min_features - n_selected, replace=False)
            particle[to_select] = True
        elif n_selected > self.max_features:
            one_indices = np.where(particle == 1)[0]
            to_remove = np.random.choice(one_indices, size=n_selected - self.max_features, replace=False)
            particle[to_remove] = False
        return particle

    def _calculate_fitness(self, X: np.ndarray, y: np.ndarray, particle: np.ndarray) -> float:
        particle_key = hash(particle.tobytes())
        if particle_key in self._fitness_cache:
            return self._fitness_cache[particle_key]
        n_selected = np.sum(particle)
        if n_selected < self.min_features or n_selected > self.max_features:
            return float('-inf')
        try:
            X_selected = X[:, particle]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            svm = SVC(kernel='rbf', random_state=42, probability=True)
            scores = []
            for train_idx, val_idx in skf.split(X_scaled, y):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                scores.append(score)
            base_score = np.mean(scores)
            stability = 1 - np.std(scores)
            feature_penalty = 1.0 - abs(n_selected - self.target_features) / (self.max_features - self.min_features)
            fitness = (0.7 * base_score + 0.2 * stability + 0.1 * feature_penalty)
            self._fitness_cache[particle_key] = fitness
            return fitness
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return float('-inf')

    def _adaptive_parameters(self, progress: float) -> Tuple[float, float, float]:
        if self.stagnation_counter > 5:
            w = min(0.9, self.w_max + 0.1 * (self.stagnation_counter / 10))
        else:
            w = self.w_max - (self.w_max - self.w_min) * progress
        c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
        c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
        return w, c1, c2

    def _velocity_confinement(self, velocities: np.ndarray) -> np.ndarray:
        v_max = 4.0
        return np.clip(velocities, -v_max, v_max)

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        best_solution = None
        best_fitness = float('-inf')
        patience = 30
        no_improve_count = 0
        print("Starting PSO optimization...")
        for iteration in range(self.max_iter):
            progress = iteration / self.max_iter
            w, c1, c2 = self._adaptive_parameters(progress)
            diversity = self._calculate_swarm_diversity()
            self.diversity_history.append(diversity)
            for i in range(self.n_particles):
                if np.random.rand() < self.mutation_rate:
                    self.particles[i] = self._adaptive_mutation(self.particles[i])
                fitness = self._calculate_fitness(X, y, self.particles[i])
                if fitness > self.pbest_fitness[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_fitness[i] = fitness
                if fitness > self.gbest_fitness:
                    self.gbest = self.particles[i].copy()
                    self.gbest_fitness = fitness
                    no_improve_count = 0
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = self.particles[i].copy()
            r1 = np.random.rand(self.n_particles, self.n_features)
            r2 = np.random.rand(self.n_particles, self.n_features)
            particles_float = self.particles.astype(float)
            pbest_float = self.pbest.astype(float)
            gbest_float = self.gbest.astype(float)
            new_velocities = (w * self.velocities + c1 * r1 * (pbest_float - particles_float) + c2 * r2 * (gbest_float - particles_float))
            self.velocities = self._velocity_confinement(new_velocities)
            probabilities = 1 / (1 + np.exp(-self.velocities))
            self.particles = np.random.rand(self.n_particles, self.n_features) < probabilities
            for i in range(self.n_particles):
                n_selected = np.sum(self.particles[i])
                if n_selected < self.min_features:
                    zero_indices = np.where(self.particles[i] == 0)[0]
                    to_select = np.random.choice(zero_indices, size=self.min_features - n_selected, replace=False)
                    self.particles[i, to_select] = True
            self.fitness_history.append(self.gbest_fitness)
            if self.gbest_fitness <= best_fitness:
                no_improve_count += 1
            else:
                no_improve_count = 0
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.gbest_fitness:.4f}, Selected features = {np.sum(self.gbest)}, Diversity = {diversity:.4f}")
            if no_improve_count >= patience:
                print(f"Early stopping at iteration {iteration}")
                break
        return best_solution, self.fitness_history, self.diversity_history

def select_features(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[List[str], List[float]]:
    print(f"Starting feature selection process...")
    print(f"Dataset shape: {X.shape}")
    print(f"Available features: {len(feature_names)}")
    bpso = BinaryPSO(
        n_particles=40,
        n_features=X.shape[1],
        max_iter=150,
        min_features=10,
        max_features=200,
        target_features=100,
        mutation_rate=0.02
    )
    best_mask, fitness_history, diversity_history = bpso.optimize(X, y)
    selected_features = [name for name, selected in zip(feature_names, best_mask) if selected]
    print("\nFeature Selection Results:")
    print(f"Selected {len(selected_features)} features")
    print(f"Final fitness score: {fitness_history[-1]:.4f}")
    print(f"Optimization completed in {len(fitness_history)} iterations")
    return selected_features, fitness_history, diversity_history

def visualize_optimization(fitness_history: List[float], diversity_history: List[float]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(fitness_history, 'b-', label='Fitness Score')
    ax1.plot(fitness_history, 'r.', alpha=0.5)
    z = np.polyfit(range(len(fitness_history)), fitness_history, 1)
    p = np.poly1d(z)
    ax1.plot(range(len(fitness_history)), p(range(len(fitness_history))), "r--", alpha=0.8, label='Trend')
    ax1.set_title('PSO Optimization Progress')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    stats_text = (f'Final Score: {fitness_history[-1]:.4f}\n'
                  f'Improvement: {fitness_history[-1] - fitness_history[0]:.4f}\n'
                  f'Iterations: {len(fitness_history)}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.plot(diversity_history, 'g-', label='Swarm Diversity')
    ax2.set_title('Swarm Diversity Over Iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Diversity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()

def train_resnet(train_loader, val_loader, num_classes=2, epochs=50):
    resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, num_classes)
    resnet_model = resnet_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(epochs):
        resnet_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    resnet_train_time = time.time() - start_time
    resnet_model.eval()
    y_pred = []
    y_prob = []
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = resnet_model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    resnet_inf_time = time.time() - start_time
    torch.cuda.empty_cache()  # Clear GPU memory
    return resnet_model, np.array(y_pred), np.array(y_prob), resnet_train_time, resnet_inf_time

def compute_and_print_metrics(y_true, y_pred, y_prob, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)  # Binary case, no multi_class needed
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

def get_pytorch_model_size(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb

class SubcategoryModel(nn.Module):
    def __init__(self, num_classes):
        super(SubcategoryModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, val_loader, epochs=100, checkpoint_path=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        best_val_acc = 0
        best_weights = None
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)  # Fixed here
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = val_correct / val_total
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.model.state_dict().copy()
            print(f"Epoch {epoch + 1}: Validation Accuracy = {val_acc:.4f}")
        self.model.load_state_dict(best_weights)
        if checkpoint_path:
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        torch.cuda.empty_cache()  # Clear GPU memory

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = image_tensor.to(device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

def extract_features_for_dataset(dataset, extractor):
    X_features = []
    y_labels = []
    feature_names = None
    failed_images = 0
    expected_length = None
    print("Extracting handcrafted features...")
    for idx, (img, label) in enumerate(dataset):
        try:
            features = extractor.extract_features(img)
            feature_vector = []
            names = []
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                    names.append(key)
                elif isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                    names.extend([f"{key}_{i}" for i in range(len(value))])
            if feature_names is None:
                feature_names = names
                expected_length = len(feature_vector)
            if len(feature_vector) != expected_length:
                print(f"Warning: Feature vector length mismatch at image {idx} (expected {expected_length}, got {len(feature_vector)})")
                continue
            X_features.append(feature_vector)
            y_labels.append(label)
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")
        except Exception as e:
            failed_images += 1
            print(f"Warning: Failed to process image {idx}: {str(e)}")
    print(f"\nFeature extraction completed: Successfully processed {len(X_features)} images, Failed {failed_images} images")
    return np.array(X_features), np.array(y_labels), feature_names if feature_names else []

def classify_image(binary_model, benign_model, malignant_model, image_path, extractor):
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return None, None
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    features = extractor.extract_features(image_tensor)
    feature_vector = []
    for key, value in features.items():
        if isinstance(value, (int, float)):
            feature_vector.append(value)
        elif isinstance(value, np.ndarray):
            feature_vector.extend(value)
    X_new = np.array([feature_vector])
    binary_pred, binary_prob = binary_model.predict(X_new, loader=DataLoader([(image_tensor, 0)], batch_size=1))
    binary_class = 'benign' if binary_pred[0] == 0 else 'malignant'
    try:
        if binary_class == 'benign':
            subcategory_pred = benign_model.predict(image_tensor)
            subcategory = benign_classes[subcategory_pred[0]]
        else:
            subcategory_pred = malignant_model.predict(image_tensor)
            subcategory = malignant_classes[subcategory_pred[0]]
        return binary_class, subcategory
    except IndexError as e:
        print(f"Error in subcategory prediction: {e}")
        return binary_class, None

if __name__ == '__main__':
    try:
        print("Starting skin cancer classification pipeline...")

        # Step 1: GLCM Extraction and Visualization
        print("\n1. Initializing GLCM Feature Extraction...")
        glcm_extractor = GLCMExtractor(distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8)
        
        print("Processing initial feature map example...")
        sample_image_path = r"D:\college - Sri eshwar\sumathi mam\datasets\skin_cancer1\Skin cancer ISIC The International Skin Imaging Collaboration\Train\malignant\basal cell carcinoma\ISIC_0025467.jpg"
        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(f"Sample image not found: {sample_image_path}")
        sample_image = Image.open(sample_image_path).convert('RGB')
        sample_tensor = transform(sample_image).unsqueeze(0)
        res_model = FeatureExtract()
        feature = res_model(sample_tensor)
        feature_map_np = feature[0, 0].detach().cpu().numpy()
        glcm_features, glcm_fig = analyze_feature_map_texture(feature_map_np, glcm_extractor)
        
        print("\nGLCM Texture Features (Example):")
        for name, value in glcm_features.items():
            print(f"{name}: {value:.4f}")
        plt.show()

        # Step 2: Single Image Feature Extraction and Visualization
        print("\n2. Processing Single Image Example...")
        image_pil = Image.open(sample_image_path).convert('RGB')
        extractor = HandcraftedFeatureExtractor(color_bins=32, fourier_points=64)
        features = extractor.extract_features(image_pil)
        fig = extractor.visualize_features(image_pil, features)
        plt.show()
        
        print("\nFeature Statistics:")
        for key, value in features.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")

        # Step 3: Feature Extraction for Dataset
        print("\n3. Extracting Handcrafted Features for Dataset...")
        X_train, y_train, feature_names = extract_features_for_dataset(binary_train_subset, extractor)
        X_val, y_val, _ = extract_features_for_dataset(binary_val_subset, extractor)
        if X_train.size == 0 or X_val.size == 0:
            raise ValueError("Feature extraction failed: No features extracted.")

        # Step 4: Feature Selection
        print("\n4. Performing Feature Selection...")
        selected_features, fitness_history, diversity_history = select_features(X_train, y_train, feature_names)
        visualize_optimization(fitness_history, diversity_history)
        print(f"Number of selected features: {len(selected_features)}")
        print("Selected features:")
        for feature in selected_features:
            print(f"- {feature}")

        # Select features
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]

        # Step 5: Train Binary Classifier from scratch
        print("\n5. Training Binary Hybrid Ensemble from Scratch...")
        input_shape = (3, 224, 224)
        binary_model = HybridEnsemble(input_shape=input_shape, num_classes=2)
        binary_model.fit(X_train_selected, y_train, X_val_selected, y_val, train_dataloader, val_dataloader, checkpoint_path=binary_checkpoint_path)

        # Step 6: Train Benign Subcategory Model from scratch
        print("\n6. Training Benign Subcategory Model from Scratch...")
        benign_model = SubcategoryModel(num_classes=len(benign_classes))
        benign_model.train_model(benign_train_dataloader, benign_test_dataloader, checkpoint_path=benign_checkpoint_path)

        # Step 7: Train Malignant Subcategory Model from scratch
        print("\n7. Training Malignant Subcategory Model from Scratch...")
        malignant_model = SubcategoryModel(num_classes=len(malignant_classes))
        malignant_model.train_model(malignant_train_dataloader, malignant_test_dataloader, checkpoint_path=malignant_checkpoint_path)

        # Step 8: Benchmarking and Validation
        print("\n8. Performing Benchmarking and Validation...")
        
        # Train XGBoost
        print("Training XGBoost...")
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train_selected, y_train)
        xgb_train_time = time.time() - start_time

        # Train ResNet-50
        print("Training ResNet-50...")
        resnet_model, resnet_pred, resnet_prob, resnet_train_time, resnet_inf_time = train_resnet(train_dataloader, val_dataloader)

        # Collect predictions
        models = {
            "SVM": (binary_model.svm, X_val_selected),
            "Random Forest": (binary_model.rf, X_val_selected),
            "CNN": (binary_model.cnn, val_dataloader),
            "XGBoost": (xgb_model, X_val_selected),
            "ResNet-50": (resnet_model, val_dataloader),
            "Hybrid Ensemble": (binary_model, (X_val_selected, val_dataloader))
        }

        predictions = {}
        probabilities = {}
        inference_times = {}

        for name, (model, data) in models.items():
            print(f"Collecting predictions for {name}...")
            start_time = time.time()
            if name in ["CNN", "ResNet-50"]:
                pred, prob = [], []
                with torch.no_grad():
                    for inputs, _ in data:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        pred.extend(predicted.cpu().numpy())
                        prob.extend(probs.cpu().numpy())
                pred = np.array(pred)
                prob = np.array(prob)
            elif name == "Hybrid Ensemble":
                pred, prob = model.predict(*data)
            else:
                pred = model.predict(data)
                prob = model.predict_proba(data)[:, 1]
            inference_times[name] = time.time() - start_time
            predictions[name] = pred
            probabilities[name] = prob

        # Compute and print metrics
        print("\nComputing Classification Metrics...")
        for name in models:
            compute_and_print_metrics(y_val, predictions[name], probabilities[name], name)

        # Compute and print computational metrics
        print("\nComputational Metrics:")
        for name in models:
            if name == "Hybrid Ensemble":
                train_time = binary_model.svm_train_time + binary_model.rf_train_time + binary_model.cnn_train_time
            elif name == "XGBoost":
                train_time = xgb_train_time
            elif name == "ResNet-50":
                train_time = resnet_train_time
            elif name == "SVM":
                train_time = binary_model.svm_train_time
            elif name == "Random Forest":
                train_time = binary_model.rf_train_time
            elif name == "CNN":
                train_time = binary_model.cnn_train_time
            else:
                train_time = 0
            inf_time = inference_times[name]
            if name in ["CNN", "ResNet-50"]:
                size = get_pytorch_model_size(models[name][0])
            else:
                size = len(pickle.dumps(models[name][0])) / (1024 * 1024)
            print(f"{name}:")
            print(f"  Training Time: {train_time:.2f} seconds")
            print(f"  Inference Time: {inf_time:.2f} seconds")
            print(f"  Model Size: {size:.2f} MB")

        # Cross-validation
        print("\n9. 5x Repeated Stratified Cross-Validation for Handcrafted Feature Models:")
        cv_models = {
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42)
        }
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        for name, model in cv_models.items():
            print(f"Performing CV for {name}...")
            scores = cross_val_score(model, X_train_selected, y_train, cv=rskf, scoring='accuracy')
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"{name}: Mean Accuracy = {mean_score:.4f} (+/- {std_score:.4f})")

        # McNemar's Test
        print("\n10. McNemar's Test: Hybrid Ensemble vs Benchmark Models:")
        ensemble_pred = predictions["Hybrid Ensemble"]
        for name in ["SVM", "Random Forest", "CNN", "XGBoost", "ResNet-50"]:
            bench_pred = predictions[name]
            table = np.zeros((2, 2))
            for i in range(len(y_val)):
                if ensemble_pred[i] == y_val[i] and bench_pred[i] == y_val[i]:
                    table[0, 0] += 1
                elif ensemble_pred[i] != y_val[i] and bench_pred[i] == y_val[i]:
                    table[0, 1] += 1
                elif ensemble_pred[i] == y_val[i] and bench_pred[i] != y_val[i]:
                    table[1, 0] += 1
                else:
                    table[1, 1] += 1
            result = mcnemar(table, exact=True)
            print(f"Hybrid Ensemble vs {name}: p-value = {result.pvalue:.4f}")

        #  classification
        print("\n11. Classifying an Example Image...")
        example_image_path = r"D:\college - Sri eshwar\sumathi mam\datasets\skin_cancer1\Skin cancer ISIC The International Skin Imaging Collaboration\Test\benign\nevus\ISIC_0000008.jpg"
        binary_class, subcategory = classify_image(binary_model, benign_model, malignant_model, example_image_path, extractor)
        if binary_class and subcategory:
            print(f"Image Classification: {binary_class}, Subcategory: {subcategory}")
        else:
            print("Failed to classify example image.")

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
    finally:
        plt.close('all')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()