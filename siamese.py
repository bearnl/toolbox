import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Model, applications
import random
from collections import defaultdict
from sklearn.metrics import pairwise_distances
import gc

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED = 2025
IMG_H, IMG_W = 192, 192
EMB_DIM = 256
DROPOUT_RATE = 0.2
MARGIN_DEPTH = 0.5
MARGIN_RGB = 0.5
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
LR_MIN = 5e-5
WEIGHT_DECAY_DEPTH = 1e-4
WEIGHT_DECAY_RGB = 1e-4
BASE_DIR = "/mnt/g/biwi"
USE_TRIPLET_LOSS = True
USE_REDUCE_LR_ON_PLATEAU = True
BODY_THICKNESS_THRESHOLD = 1500.0
CLIP_RANGE = 800.0
FOREGROUND_CLOSEST_PERCENTILE = 1
MIN_FOREGROUND_PIXELS = 100
EVAL_BATCH_SIZE = 32

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs Found: {len(gpus)}")
    except RuntimeError as e:
        print(e)

def get_augmentation_params():
    return {
        'angle': np.random.uniform(-5, 5),
        'scale': np.random.uniform(0.9, 1.0),
        'flip': np.random.rand() > 0.5,
        'noise': np.random.normal(0, 0.01, (IMG_H, IMG_W, 1))
    }

def apply_depth_augmentation(img, params):
    h, w = img.shape[:2]
    
    M = cv2.getRotationMatrix2D((w/2, h/2), params['angle'], 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    scale = params['scale']
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
    
    if params['flip']:
        img = cv2.flip(img, 1)
    
    return img

def preprocess_depth_smart(img, augment=False, aug_params=None):
    img = img.astype(np.float32)
    
    if augment and aug_params is not None:
        img = apply_depth_augmentation(img, aug_params)
    
    valid_mask = img > 0
    if np.sum(valid_mask) == 0:
        print("Warning: No valid depth pixels found in image")
        return np.zeros((IMG_H, IMG_W, 1), dtype=np.float32)
        
    valid_pixels = img[valid_mask]
    
    depth_min = np.percentile(valid_pixels, 1)
    depth_max = np.percentile(valid_pixels, 99)
    
    img_normalized = np.zeros_like(img)
    img_normalized[valid_mask] = (img[valid_mask] - depth_min) / (depth_max - depth_min + 1e-6)
    img_normalized = np.clip(img_normalized, 0.0, 1.0)
    img_normalized = (img_normalized * 2.0) - 1.0
    
    if augment and aug_params is not None:
        img_normalized = img_normalized + aug_params['noise'][:,:,0:1]
        img_normalized = np.clip(img_normalized, -1.0, 1.0)
    
    return np.expand_dims(img_normalized, -1)

def load_and_preprocess_single(path, is_rgb, augment=False, aug_params=None):
    if is_rgb:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load RGB image: {path}")
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        return img
    else:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f"Warning: Failed to load depth image: {path}")
            return np.zeros((IMG_H, IMG_W, 1), dtype=np.float32)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = preprocess_depth_smart(img, augment=augment, aug_params=aug_params)
        return img

def verify_data_quality(file_paths, labels):
    print(f"\nVerifying data quality...")
    issues = []
    
    if len(file_paths) != len(labels):
        issues.append(f"Mismatch: {len(file_paths)} files vs {len(labels)} labels")
    
    unique_labels = set(labels)
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())
    avg_samples = np.mean(list(label_counts.values()))
    
    print(f"  Identities: {len(unique_labels)}")
    print(f"  Samples per identity: min={min_samples}, max={max_samples}, avg={avg_samples:.1f}")
    
    if max_samples > 3 * min_samples:
        issues.append(f"High imbalance: max/min ratio = {max_samples/min_samples:.1f}")
    
    duplicates = len(file_paths) - len(set(file_paths))
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate file paths")
    
    if issues:
        print(f"  ⚠ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ No issues detected")
    
    return len(issues) == 0

def load_biwi_disjoint(base_dir, test_split=0.2):
    print(f"Scanning {base_dir}...")
    rgb_data = defaultdict(list)
    depth_data = defaultdict(list)
    
    search_dirs = [os.path.join(base_dir, 'Training')]
    all_identities = set()

    for search_dir in search_dirs:
        if not os.path.exists(search_dir): continue
        subjects = sorted(os.listdir(search_dir))
        for subj in subjects:
            subj_dir = os.path.join(search_dir, subj)
            if not os.path.isdir(subj_dir): continue
            all_identities.add(subj)
            for f in os.listdir(subj_dir):
                fpath = os.path.join(subj_dir, f)
                if f.lower().endswith('_rgb.jpg'):
                    rgb_data[subj].append(fpath)
                elif f.lower().endswith('_depth.pgm'):
                    depth_data[subj].append(fpath)

    all_ids = sorted(list(all_identities))
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * test_split)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    
    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError(f"Insufficient identities for train ({len(train_ids)}) or val ({len(val_ids)}). Need at least 2 each.")
    
    print(f"Identities -> Train: {len(train_ids)}, Val: {len(val_ids)}")

    def flatten_data(id_set, data_dict):
        paths, labels = [], []
        for subj in id_set:
            p_list = data_dict[subj]
            paths.extend(p_list)
            labels.extend([subj] * len(p_list))
        return paths, labels

    rgb_train = flatten_data(train_ids, rgb_data)
    rgb_val = flatten_data(val_ids, rgb_data)
    depth_train = flatten_data(train_ids, depth_data)
    depth_val = flatten_data(val_ids, depth_data)
    
    print(f"\n--- RGB Data Quality ---")
    verify_data_quality(rgb_train[0], rgb_train[1])
    verify_data_quality(rgb_val[0], rgb_val[1])
    
    print(f"\n--- Depth Data Quality ---")
    verify_data_quality(depth_train[0], depth_train[1])
    verify_data_quality(depth_val[0], depth_val[1])
    
    return (rgb_train, rgb_val, depth_train, depth_val)

class SiamesePairGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, is_rgb, shuffle=True, cache_images=True, augment=False, hard_negatives=False, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.is_rgb = is_rgb
        self.shuffle = shuffle
        self.cache_images = cache_images
        self.augment = augment
        self.hard_negatives = hard_negatives
        self.image_cache = {}
        self.embeddings = None
        self.backbone = None
        
        self.indices_by_label = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.indices_by_label[lbl].append(idx)
        self.unique_labels = [l for l in self.indices_by_label.keys() if len(self.indices_by_label[l]) > 1]
        
        if self.cache_images:
            print(f"Caching {len(self.file_paths)} images ({'RGB' if is_rgb else 'Depth'})...")
            for i, path in enumerate(self.file_paths):
                if i % 500 == 0: print(f"  {i}/{len(self.file_paths)}...", end='\r')
                self.image_cache[path] = load_and_preprocess_single(path, self.is_rgb)
            print(f"  Done.")
        self.on_epoch_end()
    
    def update_embeddings(self, backbone):
        if not self.hard_negatives:
            return
        self.backbone = backbone
        print(f"  Computing embeddings for hard negative mining...")
        if self.cache_images:
            imgs = np.array([self.image_cache[p] for p in self.file_paths])
        else:
            imgs = np.array([load_and_preprocess_single(p, self.is_rgb) for p in self.file_paths])
        self.embeddings = backbone.predict(imgs, batch_size=EVAL_BATCH_SIZE, verbose=0)
        print(f"  Done.")
        
    def __len__(self): return len(self.pairs) // self.batch_size

    def on_epoch_end(self):
        self.pairs = []
        self.pair_labels = []
        all_indices = np.arange(len(self.file_paths))
        if self.shuffle: np.random.shuffle(all_indices)
            
        for anchor_idx in all_indices:
            anchor_label = self.labels[anchor_idx]
            candidates = self.indices_by_label[anchor_label]
            if len(candidates) < 2: continue
            pos_idx = random.choice(candidates)
            while pos_idx == anchor_idx: pos_idx = random.choice(candidates)
            self.pairs.append([anchor_idx, pos_idx])
            self.pair_labels.append(1.0)
            
            if self.hard_negatives and self.embeddings is not None:
                anchor_emb = self.embeddings[anchor_idx]
                neg_candidates = []
                for neg_label in self.unique_labels:
                    if neg_label != anchor_label:
                        neg_candidates.extend(self.indices_by_label[neg_label])
                
                if len(neg_candidates) > 0:
                    neg_embs = self.embeddings[neg_candidates]
                    dists = np.sum((anchor_emb - neg_embs) ** 2, axis=1)
                    hardest_idx = neg_candidates[np.argmin(dists)]
                    neg_idx = hardest_idx
                else:
                    neg_label = random.choice(self.unique_labels)
                    while neg_label == anchor_label: neg_label = random.choice(self.unique_labels)
                    neg_idx = random.choice(self.indices_by_label[neg_label])
            else:
                neg_label = random.choice(self.unique_labels)
                while neg_label == anchor_label: neg_label = random.choice(self.unique_labels)
                neg_idx = random.choice(self.indices_by_label[neg_label])
            
            self.pairs.append([anchor_idx, neg_idx])
            self.pair_labels.append(0.0)
        
        zipped = list(zip(self.pairs, self.pair_labels))
        np.random.shuffle(zipped)
        self.pairs, self.pair_labels = zip(*zipped)
        self.pairs = np.array(self.pairs)
        self.pair_labels = np.array(self.pair_labels, dtype=np.float32)

    def __getitem__(self, index):
        indices = self.pairs[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.pair_labels[index * self.batch_size : (index + 1) * self.batch_size]
        A, B = [], []
        for (ia, ib) in indices:
            pa, pb = self.file_paths[ia], self.file_paths[ib]
            if self.cache_images:
                A.append(self.image_cache[pa])
                B.append(self.image_cache[pb])
            else:
                aug_params = get_augmentation_params() if self.augment else None
                A.append(load_and_preprocess_single(pa, self.is_rgb, augment=self.augment, aug_params=aug_params))
                B.append(load_and_preprocess_single(pb, self.is_rgb, augment=self.augment, aug_params=aug_params))
        return (np.array(A), np.array(B)), labels

class EvaluationCallback(Callback):
    def __init__(self, backbone, val_paths, val_labels, is_rgb, results_dict):
        super().__init__()
        self.backbone = backbone
        self.val_labels = np.array(val_labels)
        self.results_dict = results_dict
        self.modality = "RGB" if is_rgb else "DEPTH"
        
        self.val_imgs = None
        if not is_rgb:
            print(f"Pre-loading {self.modality} validation set...")
            self.val_imgs = np.array([load_and_preprocess_single(p, is_rgb) for p in val_paths])
        else:
            self.val_paths = val_paths
            self.is_rgb = is_rgb

    def on_epoch_end(self, epoch, logs=None):
        if self.val_imgs is not None:
            embs = self.backbone.predict(self.val_imgs, batch_size=EVAL_BATCH_SIZE, verbose=0)
        else:
            embs = []
            for i in range(0, len(self.val_paths), EVAL_BATCH_SIZE):
                batch_paths = self.val_paths[i:i+EVAL_BATCH_SIZE]
                batch_imgs = np.array([load_and_preprocess_single(p, self.is_rgb) for p in batch_paths])
                batch_embs = self.backbone.predict(batch_imgs, verbose=0)
                embs.append(batch_embs)
            embs = np.vstack(embs)

        dist_matrix = pairwise_distances(embs, metric='cosine')
        np.fill_diagonal(dist_matrix, np.inf)
        
        rank1, rank5 = 0, 0
        aps = []
        total = len(self.val_labels)
        
        for i in range(total):
            dists = dist_matrix[i]
            sorted_idx = np.argsort(dists)
            matches = (self.val_labels[sorted_idx] == self.val_labels[i])
            
            if matches[0]: rank1 += 1
            if np.any(matches[:5]): rank5 += 1
            
            num_valid = np.sum(matches)
            if num_valid > 0:
                old_recall = 0.0
                old_precision = 1.0
                ap = 0.0
                intersect_size = 0
                for j, match in enumerate(matches):
                    if match:
                        intersect_size += 1
                        recall = intersect_size / num_valid
                        precision = intersect_size / (j + 1)
                        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                        old_recall = recall
                        old_precision = precision
                aps.append(ap)
                
        r1 = rank1 / total * 100.0
        r5 = rank5 / total * 100.0
        mAP = np.mean(aps) * 100.0 if aps else 0.0
        
        emb_norms = np.linalg.norm(embs, axis=1)
        avg_norm = np.mean(emb_norms)
        std_norm = np.std(emb_norms)
        
        intra_dists = []
        inter_dists = []
        unique_labels = np.unique(self.val_labels)
        
        for label in unique_labels:
            mask = self.val_labels == label
            if np.sum(mask) > 1:
                class_embs = embs[mask]
                class_dists = pairwise_distances(class_embs, metric='cosine')
                intra_dists.extend(class_dists[np.triu_indices_from(class_dists, k=1)])
        
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                mask_i = self.val_labels == unique_labels[i]
                mask_j = self.val_labels == unique_labels[j]
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    inter_dist = pairwise_distances(embs[mask_i], embs[mask_j], metric='cosine')
                    inter_dists.extend(inter_dist.flatten())
        
        intra_mean = np.mean(intra_dists) if intra_dists else 0
        inter_mean = np.mean(inter_dists) if inter_dists else 0
        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else 0
        
        if epoch % 10 == 0 or epoch == 0:
            top_k = min(5, len(unique_labels))
            confusion = np.zeros((top_k, top_k), dtype=int)
            top_labels = unique_labels[:top_k]
            
            for i in range(total):
                if self.val_labels[i] not in top_labels:
                    continue
                dists = dist_matrix[i]
                pred_idx = np.argmin(dists)
                pred_label = self.val_labels[pred_idx]
                
                true_idx = np.where(top_labels == self.val_labels[i])[0]
                pred_idx_in_top = np.where(top_labels == pred_label)[0]
                
                if len(true_idx) > 0 and len(pred_idx_in_top) > 0:
                    confusion[true_idx[0], pred_idx_in_top[0]] += 1
            
            if epoch % 10 == 0:
                print(f"\n   Confusion (top {top_k} IDs):")
                print(f"   {confusion}")
        
        print(f" - {self.modality} | R1: {r1:.2f}% | R5: {r5:.2f}% | mAP: {mAP:.2f}% | Sep: {separation_ratio:.2f} | Norm: {avg_norm:.3f}±{std_norm:.3f}")
        
        if logs is not None:
            logs['val_rank1'] = r1
        
        if r1 > self.results_dict['best_rank1']:
            self.results_dict['best_rank1'] = r1
            self.results_dict['best_rank5'] = r5
            self.results_dict['best_mAP'] = mAP
            self.results_dict['best_separation'] = separation_ratio
            self.results_dict['best_epoch'] = epoch + 1

def get_densenet_backbone(input_shape, emb_dim, weight_decay):
    inputs = layers.Input(shape=input_shape)
    base_model = applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling='avg'
    )
    
    x = base_model(inputs)
    
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(emb_dim, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    
    return Model(inputs, outputs, name="densenet_backbone")

def make_siamese_densenet(input_shape, weight_decay):
    backbone = get_densenet_backbone(input_shape, EMB_DIM, weight_decay)
    
    a = tf.keras.Input(shape=input_shape)
    b = tf.keras.Input(shape=input_shape)
    
    feat_a = backbone(a)
    feat_b = backbone(b)
    
    def cosine_sim(feats):
        x, y = feats
        return tf.reduce_sum(x * y, axis=1, keepdims=True)
    
    distance = layers.Lambda(cosine_sim)([feat_a, feat_b])
    
    return Model(inputs=[a, b], outputs=distance), backbone

def loss_fn(margin):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        pos_loss = y_true * (1.0 - y_pred)
        neg_loss = (1.0 - y_true) * tf.nn.relu(y_pred - margin)
        
        return tf.reduce_mean(pos_loss + neg_loss)
    return loss

def triplet_loss_fn(margin):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        pos_mask = y_true > 0.5
        neg_mask = y_true < 0.5
        
        pos_sim = tf.boolean_mask(y_pred, pos_mask)
        neg_sim = tf.boolean_mask(y_pred, neg_mask)
        
        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim
        
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)
        neg_dist_expanded = tf.expand_dims(neg_dist, 0)
        
        triplet_loss = tf.nn.relu(pos_dist_expanded - neg_dist_expanded + margin)
        
        num_valid = tf.cast(tf.size(pos_sim) * tf.size(neg_sim), tf.float32)
        loss_sum = tf.reduce_sum(triplet_loss)
        
        return tf.where(num_valid > 0, loss_sum / num_valid, 0.0)
    return loss

def run_training_session(modality_name, train_p, train_l, val_p, val_l):
    tf.keras.backend.clear_session()
    gc.collect()
    
    is_rgb = (modality_name == "RGB")
    input_shape = (IMG_H, IMG_W, 3) if is_rgb else (IMG_H, IMG_W, 1)
    do_cache = not is_rgb
    margin = MARGIN_RGB if is_rgb else MARGIN_DEPTH
    weight_decay = WEIGHT_DECAY_RGB if is_rgb else WEIGHT_DECAY_DEPTH
    
    augment_data = not is_rgb
    
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING: {modality_name} (DenseNet121)")
    print(f"Cache: {do_cache} | Augment: {augment_data}")
    print(f"Margin: {margin} | Weight Decay: {weight_decay}")
    print(f"{'='*60}")
    
    train_gen = SiamesePairGenerator(train_p, train_l, BATCH_SIZE, is_rgb, shuffle=True, cache_images=do_cache, augment=augment_data, hard_negatives=augment_data)
    
    siam, back = make_siamese_densenet(input_shape, weight_decay)
    
    if USE_REDUCE_LR_ON_PLATEAU:
        optimizer = tf.keras.optimizers.Adam(LR)
    else:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(LR, EPOCHS * len(train_gen), LR_MIN)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    
    loss_function = triplet_loss_fn(margin) if USE_TRIPLET_LOSS else loss_fn(margin)
    siam.compile(optimizer=optimizer, loss=loss_function)
    
    results = {'best_rank1': 0.0, 'best_rank5': 0.0, 'best_mAP': 0.0, 'best_separation': 0.0, 'best_epoch': 0}
    
    eval_cb = EvaluationCallback(back, val_p, val_l, is_rgb, results)
    
    class HardNegativeCallback(tf.keras.callbacks.Callback):
        def __init__(self, generator, backbone, update_freq=5):
            super().__init__()
            self.generator = generator
            self.backbone = backbone
            self.update_freq = update_freq
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.update_freq == 0:
                self.generator.update_embeddings(self.backbone)
    
    hard_neg_cb = HardNegativeCallback(train_gen, back, update_freq=5) if augment_data else None
    
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"best_densenet_{modality_name.lower()}.weights.h5", 
        save_best_only=True, save_weights_only=True, monitor='loss'
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks = [eval_cb, ckpt, early_stop]
    if hard_neg_cb:
        callbacks.append(hard_neg_cb)
    
    if USE_REDUCE_LR_ON_PLATEAU:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_rank1',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=LR_MIN,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    siam.fit(train_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    return results

if __name__ == "__main__":
    (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l) = load_biwi_disjoint(BASE_DIR)
    
    depth_res = run_training_session("DEPTH", d_tr_p, d_tr_l, d_va_p, d_va_l)
    
    rgb_res = run_training_session("RGB", rgb_tr_p, rgb_tr_l, rgb_va_p, rgb_va_l)
    
    print(f"\n{'='*85}")
    print(f"FINAL COMPARISON RESULTS (DenseNet121, Up to {EPOCHS} Epochs)")
    print(f"{'='*85}")
    print(f"{'MODALITY':<10} | {'RANK-1':<10} | {'RANK-5':<10} | {'mAP':<10} | {'SEP':<8} | {'EPOCH':<6}")
    print("-" * 85)
    print(f"{'DEPTH':<10} | {depth_res['best_rank1']:.2f}%     | {depth_res['best_rank5']:.2f}%     | {depth_res['best_mAP']:.2f}%     | {depth_res['best_separation']:.2f}     | {depth_res['best_epoch']}")
    print(f"{'RGB':<10}   | {rgb_res['best_rank1']:.2f}%     | {rgb_res['best_rank5']:.2f}%     | {rgb_res['best_mAP']:.2f}%     | {rgb_res['best_separation']:.2f}     | {rgb_res['best_epoch']}")
    print("-" * 85)
    print(f"\nSEP = Separation Ratio (inter-class/intra-class distance, higher is better)")
    if USE_TRIPLET_LOSS:
        print(f"Using: Triplet Loss")
    if USE_REDUCE_LR_ON_PLATEAU:
        print(f"Using: ReduceLROnPlateau instead of CosineDecay")
