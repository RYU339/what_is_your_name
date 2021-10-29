from django.conf import settings
from gtts import gTTS
from django.shortcuts import render

# Create your views here.

def index(request):
    intro = "너의 이름은 서비스가 실행되었습니다\
    카테고리를 클릭하여 사진을 업로드해 주세요"

    tts_intro = gTTS(text=intro, lang='ko')
    tts_intro.save(settings.MEDIA_ROOT_URL + settings.MEDIA_URL + "intro.mp3")

    tts_intro = settings.MEDIA_ROOT_URL + settings.MEDIA_URL + "intro.mp3"
    context = {'tts_intro': tts_intro}

    return render(request, 'main/index.html', context)