from django.conf import settings
from tensorflow.keras import models
import numpy as np
import cv2 # opencv-python
from gtts import gTTS # google Text-To-Speech

def predict_image(path):
    # load model
    base_url = settings.MEDIA_ROOT_URL + settings.MEDIA_URL # == './media/'
    model_url = base_url + 'DL/RAMYEON_VGG19.h5'
    model = models.load_model(model_url, compile=False)

    # image resizing
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(path, img)

    img_array = np.asarray(img) # dtype이 다른 경우에만 복사한다.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # image normalization
    normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_img_array

    # image labeling
    '''
    0 bul dak
    1 cham kkae
    2 cheese
    3 jin hot
    4 jin sun
    5 neo gu ri
    6 sa ri
    7 sae u tang
    8 snack
    9 woo dong
    10 yeul
    11 yuk gae jang
    '''
    target_names = np.array(['bul dak', 'cham kkae', 'cheese', 'jin hot', 'jin sun', 'neo gu ri',
                             'sa ri', 'sae u tang', 'snack', 'woo dong', 'yeul', 'yuk gae jang'])
    
    # get index number(max)
    result = np.argmax(model.predict(data))

    # predict & sort
    pred = model.predict(data)
    # max
    predict = np.max(model.predict(data))
    # sort
    sort_predict = (-model.predict(data)).argsort()
    sort_predict = sort_predict.reshape(-1,)

    sort_value = np.sort(pred)
    sort_value = pred.reshape(-1,)

    # Predicted image labeling
    predict_result = target_names[result]

    # labeling of the rest
    rank = [] # create rank list
    rank_value = []
    # for i in sort_predict: # 오름차순
    #     rank.append(target_names[i])
    #     rank_value.append(sort_value[i])
    for i in sort_predict[::-1]: # 내림차순
        rank.append(target_names[i])
        rank_value.append(round(sort_value[i]*100, 2))
        # print(rank)

    # get image name for audio & save
    tts = gTTS(text=predict_result, lang='en')
    tts.save(base_url + "result.mp3")

    # # check cmd
    # print(result)
    # print(target_names[result])

    # rank, rank_value는 상위 3개의 순위만 가져감.
    return predict_result, predict, rank[-3:], rank_value[-3:]
