from django.db import models
import numpy as np
import matplotlib.pylab as plt
import keras, sys
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import cv2
import io, base64

# Create your models here.
graph = tf.compat.v1.get_default_graph()

# class Photo(models.Model):
#     image = models.ImageField(upload_to='photos')

#     IMAGE_SIZE = 150
#     MODEL_PATH = 'imageai/ml_models/DeepImFam.h5'
#     imagename = ['ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassE']
#     image_len = len(imagename)

#     def predict(self): 
#         model=load_model(self.MODEL_PATH)
#         global graph
#         with graph.as_default():
#             model = load_model(self.MODEL_PATH)
#             img_data = self.image.read()
#             img_bin = io.BytesIO(img_data)

#             image = Image.open(img_bin)
#             image = image.convert('L')
#             image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
#             data = np.array(image).reshape(150, 150, 1) / 255.
#             X = []
#             X.append(data)
#             X = np.array(X)

#             result=model.predict([X])[0]
#             print(result)
#             predicted=result.argmax()
#             print(predicted)
#             percentage=int(result[predicted]*100)

#             return self.imagename[predicted],percentage
#     def image_src(self):
#         with self.image.open() as img:
#             base64_img=base64.b64encode(img.read()).decode()

#             return "data:"+img.file.content_type+";base64,"+base64_img

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def generate_img(aaseq):
    amino_vector = {"A": [0.9901994150978922, 2.295540267200685], "M": [3.582951550216718, 4.812739156530262], "C": [2.1821209247191464, 2.0587249136062002], "N": [-4.010615963775219, 2.985792958513931], "D": [-2.0478801107224798, 1.433941090877615], "P": [-2.468395491102542, 4.9149795217787675], "E": [-3.73512535620498, 3.3239793280696888], "Q": [-4.993660660333668, 3.326161933736228], "F": [5.098701821977242, 4.031530693243691], "R": [-7.3860581475915605, 1.3023613325019772], "G": [-0.04937041655184403, 0.49755659172550154], "S": [-0.1744344867314273, 2.9949244748138044], "H": [-4.482150427445976, 3.988775193683627], "T": [-0.5804645706261509, 4.966191788709715], "I": [5.416442641567144, 0.9550649771681168], "V": [4.8296291314453415, 1.2940952255126026], "K": [-6.264540531214175, 3.123384691768663], "W": [-0.3256771853842271, 6.992419779369672], "L": [5.111411383247523, 2.030633810219764], "Y": [-2.0076226289776313, 6.705926586208422]}
    x_points, y_points = list(), list()
    x, y = 0, 0 # 初期位置
    x_points.append(0)
    y_points.append(0)
    for aa in aaseq:
        if not aa in amino_vector.keys(): continue
        x += amino_vector[aa][0]
        y += amino_vector[aa][1]
        x_points.append(x)
        y_points.append(y)
    size = 150
    plt.switch_backend("AGG")  
    fig = plt.figure(figsize=(size/100, size/100))
    plt.axis('off')
    plt.plot(x_points, y_points, color='k')
    #plt.show()
    img = cv2.cvtColor(np.array(fig2img(fig)), cv2.COLOR_BGR2GRAY)  # MatplotlibからImageに変換
    img = 255 - img # 画素値を反転
    # cv2.imwrite('media_local/aaimg.png', img=img)
    plt.clf()
    plt.close()
    return img.reshape(1, 150, 150, 1) / 255.



class DeepImFam(models.Model):
    aaseq = models.CharField(max_length=1000)
    IMAGE_SIZE = 150
    MODEL_PATH = 'imageai/ml_models/DeepImFam.h5'
    MODEL_PATH_SUB = 'imageai/ml_models/DeepImFam_sub.h5'
    MODEL_PATH_SUBSUB = 'imageai/ml_models/DeepImFam_subsub.h5'
    labels = ['ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassE']
    labels_sub = {0: 'ClassA_Nucleotide', 1: 'ClassA_Adrenergic', 2: 'ClassA_Peptide', 3: 'ClassA_Peptide', 4: 'ClassA_Amine', 5: 'ClassA_Peptide', 6: 'ClassD_Pheromone', 7: 'ClassA_Anaphylatoxin', 8: 'ClassA_Peptide', 9: 'ClassA_Leuko', 10: 'ClassC_BOSS', 11: 'ClassA_Peptide', 12: 'ClassA_Peptide', 13: 'ClassB_BrainSpec', 14: 'ClassA_Peptide', 15: 'ClassB_Cadherin', 16: 'ClassC_CalcSense', 17: 'ClassB_Calcitonin', 18: 'ClassA_Cannabinoid', 19: 'ClassA_Peptide', 20: 'ClassA_Peptide', 21: 'ClassB_Corticotropin', 22: 'ClassA_Amine', 23: 'ClassA_Peptide', 24: 'ClassB_EMR1', 25: 'ClassA_Peptide', 26: 'ClassC_CalcSense', 27: 'ClassA_Hormone', 28: 'ClassC_GABA', 29: 'ClassA_GRHR', 30: 'ClassA_Peptide', 31: 'ClassB_Gastric', 32: 'ClassB_Glucagon', 33: 'ClassC_GlutaMeta', 34: 'ClassA_Hormone', 35: 'ClassA_Thyro', 36: 'ClassB_GrowthHorm', 37: 'ClassA_Amine', 38: 'ClassA_Interleukin8', 39: 'ClassA_Peptide', 40: 'ClassB_Latrophilin', 41: 'ClassA_Lyso', 42: 'ClassA_Peptide', 43: 'ClassA_Peptide', 44: 'ClassA_Peptide', 45: 'ClassA_Melaton', 46: 'ClassB_Methuselah', 47: 'ClassA_Amine', 48: 'ClassA_Amine', 49: 'ClassA_Peptide', 50: 'ClassA_Peptide', 51: 'ClassA_Peptide', 52: 'ClassA_Peptide', 53: 'ClassA_Peptide', 54: 'ClassA_Amine', 55: 'ClassA_Olfactory', 56: 'ClassA_Peptide', 57: 'ClassA_Peptide', 58: 'ClassA_Peptide', 59: 'ClassB_PACAP', 60: 'ClassB_Parathyroid', 61: 'ClassC_CalcSense', 62: 'ClassA_Platelet', 63: 'ClassA_Peptide', 64: 'ClassA_Peptide', 65: 'ClassA_Prostanoid', 66: 'ClassA_Prostanoid', 67: 'ClassA_Peptide', 68: 'ClassA_Nucleotide', 69: 'ClassC_PutPher', 70: 'ClassB_Secretin', 71: 'ClassA_Amine', 72: 'ClassA_Peptide', 73: 'ClassA_Peptide', 74: 'ClassA_Peptide', 75: 'ClassA_Peptide', 76: 'ClassC_Taste', 77: 'ClassA_Peptide', 78: 'ClassA_Thyro', 79: 'ClassA_Hormone', 80: 'ClassA_Amine', 81: 'ClassA_Peptide', 82: 'ClassB_Vasocactive', 83: 'ClassA_Peptide', 84: 'ClassA_Peptide', 85: 'ClassE_cAMP'}
    labels_subsub = {0: 'Adenosine', 1: 'Adrenergic', 2: 'Adrenocorticotropic', 3: 'Adrenomedullin', 4: 'Adrenoreceptor', 5: 'Allatostatin', 6: 'AlphaFac', 7: 'Anaphylatoxin', 8: 'Angiotensin', 9: 'BLT2', 10: 'BOSS', 11: 'Bombesin', 12: 'Bradykinin', 13: 'BrainSpec', 14: 'C5A', 15: 'Cadherin', 16: 'CalcLike', 17: 'Calcitonin', 18: 'Cannabinoid', 19: 'Chemokine', 20: 'Cholecystokinin', 21: 'Corticotropin', 22: 'Dopamine', 23: 'Duffy', 24: 'EMR1', 25: 'Endothelin', 26: 'ExtraCalc', 27: 'FollicleStim', 28: 'GABA', 29: 'GRHR', 30: 'Galanin', 31: 'Gastric', 32: 'Glucagon', 33: 'GlutaMeta', 34: 'Gonadotrophin', 35: 'Growth', 36: 'GrowthHorm', 37: 'Histamine', 38: 'Interleukin8', 39: 'Kiss1', 40: 'Latrophilin', 41: 'LysoEdg2', 42: 'MelaninConc', 43: 'Melanocortin', 44: 'Melanocyte', 45: 'Melaton', 46: 'Methuselah', 47: 'MuscAcetyl', 48: 'Muscarinicacetylcholine', 49: 'Neuromedin', 50: 'NeuromedinB-U', 51: 'Neuropeptide', 52: 'NeuropeptideFF', 53: 'Neurotensin', 54: 'Octopamine', 55: 'Olfactory', 56: 'Opoid', 57: 'Orexin', 58: 'Oxytocin', 59: 'PACAP', 60: 'Parathyroid', 61: 'Pheromone', 62: 'Platelet', 63: 'Prokineticin', 64: 'Prolactin', 65: 'Prostacyclin', 66: 'Prostaglandin', 67: 'Proteinase', 68: 'Purinergic', 69: 'PutPher', 70: 'Secretin', 71: 'Serotonin', 72: 'Somatostatin', 73: 'SubstanceK', 74: 'SubstanceP', 75: 'Tachykinin', 76: 'Taste', 77: 'Thrombin', 78: 'Thyro', 79: 'Thyrotropin', 80: 'Traceamine', 81: 'UrotensinII', 82: 'Vasoactive', 83: 'Vasopressin', 84: 'Vasotocin', 85: 'cAMP'}

    def generate_img(self):
        amino_vector = {"A": [0.9901994150978922, 2.295540267200685], "M": [3.582951550216718, 4.812739156530262], "C": [2.1821209247191464, 2.0587249136062002], "N": [-4.010615963775219, 2.985792958513931], "D": [-2.0478801107224798, 1.433941090877615], "P": [-2.468395491102542, 4.9149795217787675], "E": [-3.73512535620498, 3.3239793280696888], "Q": [-4.993660660333668, 3.326161933736228], "F": [5.098701821977242, 4.031530693243691], "R": [-7.3860581475915605, 1.3023613325019772], "G": [-0.04937041655184403, 0.49755659172550154], "S": [-0.1744344867314273, 2.9949244748138044], "H": [-4.482150427445976, 3.988775193683627], "T": [-0.5804645706261509, 4.966191788709715], "I": [5.416442641567144, 0.9550649771681168], "V": [4.8296291314453415, 1.2940952255126026], "K": [-6.264540531214175, 3.123384691768663], "W": [-0.3256771853842271, 6.992419779369672], "L": [5.111411383247523, 2.030633810219764], "Y": [-2.0076226289776313, 6.705926586208422]}
        x_points, y_points = list(), list()
        x, y = 0, 0 # 初期位置
        x_points.append(0)
        y_points.append(0)
        for aa in str(self.aaseq):
            if not aa in amino_vector.keys(): continue
            x += amino_vector[aa][0]
            y += amino_vector[aa][1]
            x_points.append(x)
            y_points.append(y)
        size = 150
        plt.switch_backend("AGG")  
        fig = plt.figure(figsize=(size/100, size/100))
        plt.axis('off')
        plt.plot(x_points, y_points, color='k')
        #plt.show()
        img = cv2.cvtColor(np.array(fig2img(fig)), cv2.COLOR_BGR2GRAY)  # MatplotlibからImageに変換
        img = 255 - img # 画素値を反転
        cv2.imwrite('static/aaimg.png', img=img)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        image_png = buffer.getvalue()
        graph = base64.b64encode(image_png)
        graph = graph.decode('utf-8')
        buffer.close()
        plt.clf()
        plt.close()
        return img.reshape(1, 150, 150, 1), graph

    def predict(self): 
        # モデル読み込み
        model = load_model(self.MODEL_PATH)
        model_sub = load_model(self.MODEL_PATH_SUB)
        model_subsub = load_model(self.MODEL_PATH_SUBSUB)

        # 画像の生成
        aaimg_data, graph = self.generate_img()

        # 予測
        pred = model.predict(aaimg_data)[0]
        pred_label = np.argmax(pred)
        pred_sub = model_sub.predict(aaimg_data)[0]
        pred_sub_label = np.argmax(pred_sub)
        pred_subsub = model_subsub.predict(aaimg_data)[0]
        pred_subsub_label = np.argmax(pred_subsub)
        percentage = int(pred[pred_label] * 100)
        percentage_sub = int(pred_sub[pred_sub_label] * 100)
        percentage_subsub = int(pred_subsub[pred_subsub_label] * 100) 
        return self.labels[pred_label], percentage, self.labels_sub[pred_sub_label], percentage_sub, self.labels_subsub[pred_subsub_label], percentage_subsub, graph
        