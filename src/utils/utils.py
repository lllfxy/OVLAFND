import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
def clipdata2gpu(batch):
    # print(f"DEBUG: batch length is {len(batch)}")  # 打印 batch 的长度
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda(),
        'clip_image':batch[5].cuda(),
        'clip_text': batch[6].cuda(),
        'multi_category':batch[7].cuda(),
        # 'content_neg': batch[8].cuda(),  # 新增这一行
        # 'enhanced': batch[9].cuda(),
        # # === [必须新增这一行] ===
        # 'llm_topic_embs': batch[10].cuda()
    }
    # 关键修复：解包新增的 content_neg (idx=8) 和 enhanced (idx=9)
    if len(batch) > 8:
        batch_data['clip_content_neg'] = batch[8].cuda()
    if len(batch) > 9:
        batch_data['enhanced'] = batch[9].cuda()
    # 检查 batch 的长度是否大于 10
    if len(batch) > 10:
        batch_data['llm_topic_embs'] = batch[10].cuda()
    else:
        # 兜底机制：如果读取到了旧的缓存数据，临时给一个 512 维的 0 向量，保证不报错
        batch_size = batch[0].size(0)
        batch_data['llm_topic_embs'] = torch.zeros(batch_size, 512).cuda()
        # print("\n[Warning] ⚠️ 读取到了旧版本的数据缓存，未找到 llm_topic_embs，已用全 0 向量兜底！为了保证训练效果，请务必删除旧的 Dataset 缓存文件！")
    return batch_data
def data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda()
    }
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def metricsTrueFalse(y_true, y_pred, category, category_dict):
    y_GT = y_true
    metricsTrueFalse = metrics(y_true, y_pred, category, category_dict)
    fake = {}
    real = {}
    THRESH = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0]*9, [0]*9, [0]*9, [0]*9
    realnews_sum, fakenews_sum = [0] * 9, [0] * 9
    y_pred_probs = y_pred.copy() if isinstance(y_pred, list) else list(y_pred)

    for thresh_idx, thresh in enumerate(THRESH):
        # 每次使用原始概率进行判断，生成临时的 0/1 列表
        y_pred_binary = [0 if p < thresh else 1 for p in y_pred_probs]

        for idx in range(len(y_pred_binary)):
            if y_GT[idx] == 1:
                # FAKE NEWS RESULT
                fakenews_sum[thresh_idx] += 1
                if y_pred_binary[idx] == 0:
                    fakenews_FN[thresh_idx] += 1
                    realnews_FP[thresh_idx] += 1
                else:
                    fakenews_TP[thresh_idx] += 1
                    realnews_TN[thresh_idx] += 1
            else:
                # REAL NEWS RESULT
                realnews_sum[thresh_idx] += 1
                if y_pred_binary[idx] == 1:
                    realnews_FN[thresh_idx] += 1
                    fakenews_FP[thresh_idx] += 1
                else:
                    realnews_TP[thresh_idx] += 1
                    fakenews_TN[thresh_idx] += 1

    val_accuracy, real_accuracy, fake_accuracy, real_precision, fake_precision = [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9
    real_recall, fake_recall, real_F1, fake_F1 = [0] * 9, [0] * 9, [0] * 9, [0] * 9
    for thresh_idx, _ in enumerate(THRESH):
        total_sum = realnews_TP[thresh_idx] + realnews_TN[thresh_idx] + realnews_FP[thresh_idx] + realnews_FN[
            thresh_idx]
        val_accuracy[thresh_idx] = (realnews_TP[thresh_idx] + realnews_TN[thresh_idx]) / max(1, total_sum)
        real_accuracy[thresh_idx] = (realnews_TP[thresh_idx]) / max(1, realnews_sum[thresh_idx])
        fake_accuracy[thresh_idx] = (fakenews_TP[thresh_idx]) / max(1, fakenews_sum[thresh_idx])
        real_precision[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FP[thresh_idx]))
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx]))
        real_recall[thresh_idx] = realnews_TP[thresh_idx]/max(1,(realnews_TP[thresh_idx]+realnews_FN[thresh_idx]))
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(1,(fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx]))
        real_F1[thresh_idx] = 2*(real_recall[thresh_idx]*real_precision[thresh_idx])/max(1,(real_recall[thresh_idx]+real_precision[thresh_idx]))
        fake_F1[thresh_idx] = 2 * (fake_recall[thresh_idx] * fake_precision[thresh_idx]) / max(1,(fake_recall[thresh_idx] + fake_precision[thresh_idx]))
    fake['precision'] =fake_precision[0]
    fake['recall'] =fake_recall[0]
    fake['F1'] =fake_F1[0]
    real['precision'] =real_precision[0]
    real['recall'] =real_recall[0]
    real['F1'] =real_F1[0]
    metricsTrueFalse['real']=real
    metricsTrueFalse['fake'] = fake
    return metricsTrueFalse

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            # 修复：roc_auc_score 返回的是 float，不需要 .tolist()
            metrics_by_category[c] = {
                'auc': round(roc_auc_score(res['y_true'], res['y_pred']), 4)
            }
        except ValueError:
            pass

    try:
        # 修复：roc_auc_score 返回的是 float，不需要 .round(4).tolist()
        metrics_by_category['auc'] = round(roc_auc_score(y_true, y_pred, average='macro'), 4)
    except ValueError:
        pass
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred)

    for c, res in res_by_category.items():
        # precision, recall, fscore, support = precision_recall_fscore_support(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), zero_division=0)
        metrics_by_category[c] = {
            'precision': round(precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                         average='macro'), 4),
            'recall': round(recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                   average='macro'), 4),
            'fscore': round(f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro'),
                4),

            #'auc': metrics_by_category[c]['auc'],
            #'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
            'acc': round(accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)), 4),

        }
    return metrics_by_category


class Recorder():
    def __init__(self, early_step):
        self.max = {'metric': 0.0}
        self.cur = {'metric': 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        return self.judge()

    def judge(self):
        current_f1 = self.cur['metric']
        best_f1 = self.max['metric']

        # 严格判断 F1 是否提升
        if current_f1 > best_f1:
            print(f"🌟 [Recorder] 突破记录！Validation F1 从 {best_f1:.4f} 提升至 {current_f1:.4f}！")
            print(f"🌟 [Recorder] 正在保存当前 Epoch ({self.curindex}) 的最佳权重...")
            self.max = self.cur
            self.maxindex = self.curindex
            return 'save'

        else:
            patience_left = self.early_step - (self.curindex - self.maxindex)
            print(
                f"📉 [Recorder] Validation F1 ({current_f1:.4f}) 未超越最高纪录 ({best_f1:.4f} @ Epoch {self.maxindex})。剩余耐心值: {patience_left}")

            # 触发早停
            if self.curindex - self.maxindex >= self.early_step:
                print(f"🛑 [Recorder] 连续 {self.early_step} 个 Epoch 未提升，触发 Early Stopping！")
                self.showfinal()
                return 'esc'
            else:
                return 'continue'

    def showfinal(self):
        print("🏆 [Recorder] 最终锁定的历史最高记录: F1 =", self.max['metric'])