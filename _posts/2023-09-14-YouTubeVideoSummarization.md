---
layout: post
title: "YouTube Video Summary"
subtitle: "YouTube Video Summarization Using Hugging Face Transformers & Whisper ASR"
date: 2023-09-14
background: '/img/posts/YouTube/YouTubePic.jpg'

#make sure to swicth image path to foward slashes if using windows
#BrettNeubeck.github.io\img\posts\311-forecasting\Buffalo311Logo2.jpg
---

### Table of Contents

- [Summary](#summary)
- [Imports](#imports)
- [Model Selection](#model)
- [YouTube Meta Data Processing](#meta)
- [Audio Trimming](#trim)
- [Transcribing Audio Stream](#trans)
- [Summarize Transcribed Text](#sum)
- [Example Of Shorted Summarized Text](#short)
- [Conclusion](#conclusion)

### Summary
<a id='summary'></a>

Recently, I was tasked with watching a YouTube video and discussing it in a graduate class. Inspired by the idea of automating this process, I embarked on a project to leverage Python and machine learning techniques to summarize YouTube video transcripts. This project showcases how to utilize Hugging Face Transformers and Whisper ASR to achieve this task.

**Project Goals:**
The primary goal of this project was to develop a Python script capable of summarizing YouTube video transcripts automatically. To achieve this, I followed a series of steps, including package installation, data retrieval, audio processing, ASR (Automatic Speech Recognition), and NLP (Natural Language Processing) for summarization.

**Project Steps:**

1. **Package Installation and Import:**
   - The project begins with the installation of necessary Python packages.
   - Key libraries include Hugging Face Transformers for NLP and PyTube for YouTube data extraction.

2. **Data Retrieval:**
   - The project involves selecting a YouTube video to summarize.
   - The PyTube library is used to extract relevant information about the video.

3. **Audio Processing:**
   - The audio stream from the selected YouTube video is downloaded.
   - If required, audio splicing (similar to sampling a record) is performed to extract the relevant portion of the audio.

4. **Automatic Speech Recognition (ASR):**
   - The Whisper ASR model, is utilized to transcribe the audio feed.
   - Whisper converts the spoken words in the video into text, enabling further analysis.

5. **Text Summarization with NLP:**
   - Using NLP techniques, the transcribed text is summarized.
   - Hugging Face Transformers' NLP capabilities are employed to achieve this summarization.

**Project Outcome:**
The end result of this project is a Python script that can take a YouTube video, transcribe its audio content, and then generate a concise textual summary. This automated summarization process significantly simplifies the task of reviewing and discussing YouTube videos, making it more efficient and accessible.


```python
 !pip install git+https://github.com/openai/whisper.git -q
 !sudo apt update && sudo apt install ffmpeg -q
 !pip install pytube
 !pip install transformers
```

```python
!nvidia-smi
```

    Thu Sep 14 17:43:11 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   35C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    

### Imports
<a id='imports'></a>


```python
import whisper
from pytube import YouTube
import datetime
import pprint
from transformers import pipeline
```

### Model Selection
<a id='model'></a>

###### Whisper ASR models have the following options <br>
- tiny, base, small, medium, large


```python
# initialize whisper asr model
model = whisper.load_model('medium')
```

    100%|█████████████████████████████████████| 1.42G/1.42G [01:07<00:00, 22.6MiB/s]
    

### YouTube Meta Data Processing
<a id='meta'></a>


```python
# select youtube video
youtube_video_url = 'https://www.youtube.com/watch?v=hVimVzgtD6w'
youtube_video = YouTube(youtube_video_url)
```


```python
# print youtube video title from link
youtube_video.title
```




    "The best stats you've ever seen | Hans Rosling"




```python
# list all meta data in youtube video
dir(youtube_video)
```




    ['__class__',
     'streams',
     'thumbnail_url',
     'title',
     'use_oauth',
     'vid_info',
     'video_id',
     'views',
     'watch_html',
     'watch_url']




```python
# look for audio streams / focus on audio / audio streams will be smaller to download
youtube_video.streams
```




    [<Stream: itag="17" mime_type="video/3gpp" res="144p" fps="8fps" vcodec="mp4v.20.3" acodec="mp4a.40.2" progressive="True" type="video">, <Stream: itag="18" mime_type="video/mp4" res="360p" fps="30fps" vcodec="avc1.42001E" acodec="mp4a.40.2" progressive="True" type="video">, <Stream: itag="133" mime_type="video/mp4" res="240p" fps="30fps" vcodec="avc1.4d400d" progressive="False" type="video">, <Stream: itag="242" mime_type="video/webm" res="240p" fps="30fps" vcodec="vp9" progressive="False" type="video">, <Stream: itag="160" mime_type="video/mp4" res="144p" fps="30fps" vcodec="avc1.4d400c" progressive="False" type="video">, <Stream: itag="278" mime_type="video/webm" res="144p" fps="30fps" vcodec="vp9" progressive="False" type="video">, <Stream: itag="139" mime_type="audio/mp4" abr="48kbps" acodec="mp4a.40.5" progressive="False" type="audio">, <Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2" progressive="False" type="audio">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus" progressive="False" type="audio">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus" progressive="False" type="audio">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus" progressive="False" type="audio">]




```python
# run for loop to find streams
for stream in youtube_video.streams:
  print(stream)
```

    <Stream: itag="17" mime_type="video/3gpp" res="144p" fps="8fps" vcodec="mp4v.20.3" acodec="mp4a.40.2" progressive="True" type="video">
    <Stream: itag="18" mime_type="video/mp4" res="360p" fps="30fps" vcodec="avc1.42001E" acodec="mp4a.40.2" progressive="True" type="video">
    <Stream: itag="133" mime_type="video/mp4" res="240p" fps="30fps" vcodec="avc1.4d400d" progressive="False" type="video">
    <Stream: itag="242" mime_type="video/webm" res="240p" fps="30fps" vcodec="vp9" progressive="False" type="video">
    <Stream: itag="160" mime_type="video/mp4" res="144p" fps="30fps" vcodec="avc1.4d400c" progressive="False" type="video">
    <Stream: itag="278" mime_type="video/webm" res="144p" fps="30fps" vcodec="vp9" progressive="False" type="video">
    <Stream: itag="139" mime_type="audio/mp4" abr="48kbps" acodec="mp4a.40.5" progressive="False" type="audio">
    <Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2" progressive="False" type="audio">
    <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus" progressive="False" type="audio">
    <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus" progressive="False" type="audio">
    <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus" progressive="False" type="audio">
    


```python
# find all audio streams
streams = youtube_video.streams.filter(only_audio=True)
streams
```




    [<Stream: itag="139" mime_type="audio/mp4" abr="48kbps" acodec="mp4a.40.5" progressive="False" type="audio">, <Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2" progressive="False" type="audio">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus" progressive="False" type="audio">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus" progressive="False" type="audio">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus" progressive="False" type="audio">]




```python
# .first calls first stream in streams
stream = streams.first()
stream
```




    <Stream: itag="139" mime_type="audio/mp4" abr="48kbps" acodec="mp4a.40.5" progressive="False" type="audio">




```python
# download audio stream and save name
# will download to google colab
stream.download(filename='HansRosling.mp4')
```




    '/content/HansRosling.mp4'



### Audio Stream Trimming
<a id='trim'></a>


```python
# process audio using ffmpeg
# ! runs command line commands
# trim audio like slicing a beat
# 23 seconds is when audio began
# 1193 seconds is when audio stopped
!ffmpeg -ss 23 -i HansRosling.mp4 -t 1193 HansRoslingTrimmed.mp4

# make sure to click refresh on file window to show new trimmed file.

```

    ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers

    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'HansRosling.mp4':
      Metadata:
        major_brand     : dash
        minor_version   : 0
        compatible_brands: iso6mp41
        creation_time   : 2018-10-22T04:59:48.000000Z
      Duration: 00:20:35.77, start: 0.000000, bitrate: 47 kb/s
      Stream #0:0(eng): Audio: aac (HE-AAC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 0 kb/s (default)
        Metadata:
          creation_time   : 2018-10-22T04:59:48.000000Z
          handler_name    : SoundHandler
          vendor_id       : [0][0][0][0]
    Stream mapping:
      Stream #0:0 -> #0:0 (aac (native) -> aac (native))
    Press [q] to stop, [?] for help
    Output #0, mp4, to 'HansRoslingTrimmed.mp4':
      Metadata:
        major_brand     : dash
        minor_version   : 0
        compatible_brands: iso6mp41
        encoder         : Lavf58.76.100
      Stream #0:0(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
        Metadata:
          creation_time   : 2018-10-22T04:59:48.000000Z
          handler_name    : SoundHandler
          vendor_id       : [0][0][0][0]
          encoder         : Lavc58.134.100 aac
    size=   18923kB time=00:19:52.99 bitrate= 129.9kbits/s speed=  60x    
    video:0kB audio:18721kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 1.077332%
    [1;36m[aac @ 0x593f08700040] [0mQavg: 226.956
    

### Transcribing Audio Stream
<a id='trans'></a>


```python
# save a timestamp before transcription
t1 = datetime.datetime.now()
print(f'started at  {t1}')

# do the transcript using whisper
output = model.transcribe('HansRoslingTrimmed.mp4')

#show time elapsed after transcription is complete
t2 = datetime.datetime.now()
print(f'ended at {t2}')
print(f'time elapses: {t2 - t1}')
```

    started at  2023-09-14 17:45:24.140809
    ended at 2023-09-14 17:48:25.742648
    time elapses: 0:03:01.601839
    


```python
# pretty print the output dictionary to inspect its structure
pprint.pprint(output)
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
                              912,
                              1605,
                              2744,
                              510,
                              294,
                              50746]},
                  {'avg_logprob': -0.17825694023808347,
                   'compression_ratio': 1.5320197044334976,
                   'end': 344.32000000000005,
                   'id': 99,
                   'no_speech_prob': 0.3483344614505768,
                   'seek': 32972,
                   'start': 337.36,
                   'temperature': 0.0,
                   'text': ' Vietnam, 19, 2003 as in United States, 1974 by the '
                           'end of the war.',
                  


```python
# access the transcribed text using the appropriate key
transcribed_text = output['text']  # 'text' is the key
print(transcribed_text)  # print the transcribed text
```

     About ten years ago, I took on the task to teach global development to Swedish undergraduate students. That was after having spent about 20 years together with African institutions studying hunger in Africa. So I was sort of expected to know a little about the world. And I started in our medical university, Karolinska Institute, an undergraduate course called Global Health. But when you get that opportunity, you get a little nervous. I thought, these students coming to us actually have the highest grade you can get in Swedish college system. So I thought maybe they know everything I'm going to teach them about. So I did a pre-test when they came. And one of the questions from which I learned a lot was this one. Which country has the highest child mortality of these five pairs? And I put them together so that in each pair of country, one has twice the child mortality of the other. And this means that it's much bigger the difference than the uncertainty of the data. I won't put you at a test here, but it's Turkey, which is highest there, Poland, Russia, Pakistan, and South Africa. And these were the results of the Swedish students. I did it so I got a confidence interval, which was pretty narrow. And I got happy, of course. At one point, eight right answers out of five possible. That means that there was a place for a professor of international health and for my course. But one late night when I was compiling the report, I really realized my discovery. I have shown that Swedish top students know statistically significantly less about the world than the chimpanzees. Because the chimpanzee would score half right. If I gave them two bananas with Sri Lanka and Turkey, they would be right half of the cases. But the students are not there. The problem for me was not ignorance. It was preconceived ideas. I did also an unethical study of the professors of the Karolinska Institute that hands out the Nobel Prize in medicine, and they are on par with the chimpanzee there. So this is where I realized that there was really a need to communicate because the data of what's happening in the world and the child health of every country is very well aware. So we did this software which displays it like this. Every bubble here is a country. This country over here is China. This is India. The size of the bubble is the population. And on this axis here, I put fertility rate. Because my students, what they said when they looked upon the world, and I asked them, what do you really think about the world? Well, I first discovered that the textbook was tinted mainly. And they said the world is still we and them. And we is Western world and them is third world. And what do you mean with Western world? I said, well, that's long life in small family and third world is short life in large family. So this is what I could display here. I put fertility rate here, number of children per woman, one, two, three, four, up to about eight children per woman. We have very good data since 1962, 1960 about, on the size of families in all countries. The arrow margin is narrow. Here I put life expectancy at birth from 30 years in some countries up to about 70 years. And 1962, there was really a group of countries here that was industrialized countries. And they had small families and long lives. And these were the developing countries. They had large families and they had relatively short lives. Now, what has happened since 1962? We want to see the change. Are the students right? It's still two types of countries. Or have these developing countries got smaller families and they live here? Or have they got longer lives and live up there? Let's see. We start the world and this is all UN statistic that has been available. Here we go. Can you see that? It's China. They're moving against better health. They're improving their, all the green Latin American countries. They are moving towards smaller families. Your yellow ones here are the Arabic countries and they get larger families, but they no longer life, but not larger families. The Africans are the green down here. They still remain here. This is India. Indonesia is moving on pretty fast. In the eighties here, you have Bangladesh still among the African countries there, but not Bangladesh. It's a miracle that happens in the eighties. The imams start to promote family planning and they move up into that corner. And in nineties, we have the terrible HIV epidemic that takes down the life expectancy of the African countries and all the rest of the world moves up into the corner where we have long lives and small family and we have a completely new world. Let me make a comparison directly between United States of America and Vietnam. 1964, America had small families and long life. Vietnam had large families and short lives. And this is what happens. The data during the war indicate that even with all the death, there was an improvement of life expectancy. By the end of the year, the family planning started in Vietnam and they went for smaller families and the United States up there is getting for longer life, keeping family size. And in the eighties now, they give up communist planning and they go for market economy and it moves faster even than social life. And today we have in Vietnam the same life expectancy and the same family size here in Vietnam, 19, 2003 as in United States, 1974 by the end of the war. I think we all, if we don't look in the data, we underestimate the tremendous change in Asia, which was in social change before we saw the economical change. So let's move over to another way here in which we could display the distribution in the world of the income. This is the world distribution of income of people, $1, $10 or $100 per day. There's no gap between rich and poor any longer. This is a myth. There's a little hump here, but there are people all the way. And if we look where the income ends up, the income, this is 100% of world's annual income and the richest 20%, they take out of that about 74% and the poorest 20%, they take about 2%. And this shows that the concept developing countries is extremely doubtful. We sort of think about aid, like these people here giving aid to these people here. But in the middle, we have most of world population and they have now 24% of the income. We heard it in other forms. And who are these? Where are the different countries? I can show you Africa. This is Africa. 10% of world population, most in poverty. This is OECD, the rich country, the country club of the UN. They are over here on this side and quite an overlap between Africa and OECD. And this is Latin America. It has everything on this earth from the poorest to the richest in Latin America. And on top of that, we can put East Europe, we can put East Asia and we could South Asia. And how did it look like if we go back in time to about 1970? Then there was more of a hump. And we have most who lived in absolute poverty were Asians. The problem in the world was the poverty in Asia. And if I now let the world move forward, you will see that while population increase, there are hundreds of millions in Asia getting out of poverty and some others get into poverty. And this is the pattern we have today. And the best projection from the World Bank is that this will happen. And we will not have a divided world. We have most people in the middle. Of course, it's a logarithmic scale here. But our concept of economy is growth with percent. We look upon it as a possibility of percent increase. If I change this and I take GDP per capita instead of family income and I turn these individual data into regional data of gross domestic product and I take the regions down here, the size of the bubble is still the population and you have the OECD there and you have sub-Saharan Africa there and we take off the Arab states there coming both from Africa and from Asia and we put them separately and we can expand this axis and I can give it a new dimension here by adding the social values there, child survival. Now I have money on that axis and I have the possibility of children to survive there. In some countries, 99.7% of children survive to five years of age, others only 70. And here it seems that there is a gap between OECD, Latin America, East Europe, East Asia, Arab states, South Asia and sub-Saharan Africa. The linearity is very strong between child survival and money. But let me split sub-Saharan Africa. Health is there and better health is up there. I can go here and I can split sub-Saharan Africa into its countries and when it bursts, the size of East Country bubble is the size of the population. Sierra Leone down there, Mauritius up there. Mauritius was the first country to get away with trade barriers and they could sell their sugar, they could sell their textiles on equal terms as the people in Europe and North America. There's a huge difference between Africa and Ghana is here in the middle. In Sierra Leone, humanitarian aid, here in Uganda, development aid, here time to invest, there you can go for holiday. It's a tremendous variation within Africa which we very often make that it's equal everything. I can split South Asia here, India is the big bubble in the middle but huge difference between Afghanistan and Sri Lanka. And I can split Arab states, how are they? Same climate, same culture, same religion, huge difference even between neighbors. Yemen civil war, United Arab Emirates money which was quite equally and well used, not as the myth is and that includes all the children of the foreign workers who are in the country. Data is often better than you think. Many people say that data is bad. There is an uncertainty margin but we can see the difference here, Cambodia, Singapore. The differences are much bigger than the weakness of the data. East Europe, Soviet economy for a long time but they come out after 10 years very, very differently. And there is Latin America. Today we don't have to go to Cuba to find a healthy country in Latin America. Chile will have a lower child mortality than Cuba within some few years from now. And here we have high income countries in OECD and we get the whole pattern here of the world which is more or less like this. And if we look at it, how it looks the world, in 1960 it starts to move. 1960, this is Mao Tse Tung. He brought health to China and then he died and then Deng Xiaoping came and brought money to China and brought them into the mainstream again. And we have seen how countries move in different directions like this. So it's sort of difficult to get an example country which shows the pattern of the world. But I would like to bring you back to about here at 1960. And I would like to compare South Korea, which is this one, with Brazil, which is this one. The label went away for me here. And I would like to compare Uganda, which is there. And I can run it forward like this. And you can see how South Korea is making a very, very fast advancement whereas Brazil is much slower. And if we move back again here and we put on trails on them like this, you can see again that the speed of development is very, very different. And the countries are moving more or less in the same rate as money and health. But it seems you can move much faster if you are healthy first than if you are wealthy first. And to show that you can put on the way of United Arab Emirates, they came from here, a mineral country. They catch all the oil, they got all the money, but health cannot be bought at the supermarket. You have to invest in health. You have to get kids into schooling. You have to train health staff. You have to educate the population. And Sheikh Zayed did that in a fairly good way. In spite of falling oil prices, he brought this country up here. So we got a much more mainstream appearance of the world where all countries tend to use their money better than they used in the past. Now this is more or less if you look at average data of the countries. They are like this. Now that's dangerous to use average data because there's such a lot of difference within countries. So if I go and look here, we can see that Uganda, that today is where South Korea was 1960. If I split Uganda, there's quite a difference within Uganda. These are the quintiles of Uganda. The richest 20% of Ugandans are there. The poorest are down there. If I split South Africa, it's like this. And if I go down and look at Nigeria, where there was such a terrible famine lastly, it's like this. The 20% poorest of Nigeria is out here and the 20% richest of South Africa is there. And yet we tend to discuss on what solutions there should be in Africa. Everything in this world exists in Africa. And you can't discuss universal access to HIV for that quintile up here with the same strategy as down here. The improvement of the world must be highly contextualized. And it's not relevant to have it on regional level. We must be much more detailed. We find that students get very excited when they can use this. And even more policymakers and the corporate sectors would like to see how the world is changing. Now, why doesn't this take place? Why are we not using the data we have? We have data in the United Nations, in the national statistical agencies, and in universities and other non-governmental organizations. Because the data is hidden down in the databases. The public is there and the internet is there, but we have still not used it effectively. All that information we saw changing in the world does not include publicly funded statistics. There are some web pages like this, you know. But they take some nourishment down from the databases. But people put prices on them. Stupid passwords and boring statistics. And this won't work. So what is needed? We have the databases. It's not a new database you need. We have wonderful design tools and more and more are added up here. So we started a non-profit venture which we called Linking Data to Design. We call it GapMinder from London Underground where they warn you, mind the gap. So we thought GapMinder was appropriate. And we started to write software which could link the data like this. And it wasn't that difficult. It took some person years. And we have produced animations. You can take a data set and put it there. We are liberating UN data. Some few UN organisations, some countries accept that their databases can go out on the world. But what we really need is of course a search function. A search function where we can copy the data up to a searchable format and get it out in the world. And what do we hear when we go around? I've done anthropology on the main statistical units. Everyone says it's impossible. This can't be done. The information is so peculiar and detailed. So that cannot be searched as other can be searched. We cannot give the data free to the students, free to the entrepreneurs of the world. But this is what we would like to see, isn't it? The publicly funded data is down here and we would like flowers to grow out on the net. And one of the crucial points is to make them searchable and then people can use the different design tool to animate it there. And I have a pretty good news for you. I have a good news that the present new head of UN statistic, he doesn't say it's impossible. He only says we can't do it. And that's a quite clever guy. So we can see a lot happening in data in the coming years. We will be able to look at income distributions in completely new ways. This is the income distribution of China 1970. This is the income distribution of the United States 1970. Almost no overlap. Almost no overlap. And what has happened? What has happened is this. The China is growing. It's not so equal any longer. And it's appearing here overlooking the United States almost like a ghost, isn't it? It's pretty scary. But I think it's very important to have all this information. We need really to see it. And instead of looking at this, I would like to end up by showing the internet users per 1000. In this software, we access about 500 variables from all the countries quite easily. It takes some time to change for this. But on the accesses, you can quite easily get any variable you would like to have. And the thing would be to get up the databases free, to get them searchable, and with a second click to get them into the graphic formats where you can instantly understand them. Now the statisticians doesn't like it because they say that this will not show the reality. We have to have statistical analytical methods. But this is hypothesis generating. I end now with a world where the internet are coming. The number of internet users are going up like this. This is the GDP per capita. And it's a new technology coming in. But amazingly how well it fits to the economy of the countries. That's why the $100 computer will be so important. But it's a nice tendency. It's as if the world is flattening off, isn't it? These countries are lifting more than the economy. And it will be very interesting to follow this over the year as I would like you to be able to do with all the publicly funded data. Thank you very much. What if great ideas weren't cherished? What if they carried no importance? Or held no value?
    


```python
# Extract and concatenate the transcribed text
result = " ".join([segment['text'] for segment in output['segments']])

# Print the length of the concatenated text
print(f"Length of transcribed text: {len(result)}")

# Print the transcribed text (uncomment the line below to print the text)
# print(result)

```

    Length of transcribed text: 17564
    

### Summarize Transcribed Text
<a id='sum'></a>


```python
# initialize hugging face pipeline
summarizer = pipeline('summarization')
```


```python
# run chuncks thru hugging face pipline
num_iters = int(len(result)/1000)
summarized_text = []
for i in range(0, num_iters + 1):
  start = 0
  start = i * 1000
  end = (i + 1) * 1000
  print("input text \n" + result[start:end])
  out = summarizer(result[start:end])
  out = out[0]
  out = out['summary_text']
  print("Summarized text\n"+out)
  summarized_text.append(out)

#print(summarized_text)
```
*Example**

    input text 
     About ten years ago, I took on the task to teach global development to Swedish undergraduate  students.  That was after having spent about 20 years together with African institutions studying  hunger in Africa.  So I was sort of expected to know a little about the world.  And I started in our medical university, Karolinska Institute, an undergraduate course called  Global Health.  But when you get that opportunity, you get a little nervous.  I thought, these students coming to us actually have the highest grade you can get in Swedish  college system.  So I thought maybe they know everything I'm going to teach them about.  So I did a pre-test when they came.  And one of the questions from which I learned a lot was this one.  Which country has the highest child mortality of these five pairs?  And I put them together so that in each pair of country, one has twice the child mortality  of the other.  And this means that it's much bigger the difference than the uncertainty of the data.  I w
    Summarized text
     Swedish professor teaches global development to students in his university's Global Health course . He says one of the questions from which he learned a lot was which country has the highest child mortality of these five pairs . "In each pair of countries, one has twice the child mortality  of the other," he says .
   

```python
# print length of summarized text
len(str(summarized_text))
```




    6072




```python
# print summarized text
str(summarized_text)
```




    '[\' Swedish professor teaches global development to students in his university\\\'s Global Health course . He says one of the questions from which he learned a lot was which country has the highest child mortality of these five pairs . "In each pair of countries, one has twice the child mortality  of the other," he says .\', \' Swedish students know statistically less about the world than chimpanzees, professor says . Turkey, Poland, Russia, Pakistan, and South Africa are the highest countries in the world, he says . Professor: "The problem for me was not ignorance.  It was preconceived ideas.  I did also an unethical study of the professors of the Karolinska Institute"\', \' Every bubble here is a country.  This country over here is China. This is India.  The size of the bubble is the population.  And on this axis here, I put fertility rate. r with the chimpanzee there.  So this is where I realized that there was really a need to communicate because the data .\', \' We have very good data since 1962, 1960 about, on, on the size of families in all countries . Life expectancy at birth at birth from 30 years in some countries up to about 70 years . In 1962, there was really a group of countries here that was industrialized countries . And these were the developing countries.  They had large families and they had relatively short lives. Now, what has happened since 1962?\', \' In the eighties here, you have Bangladesh still among the African countries there, but not Bangladesh . In nineties, we have the terrible HIV epidemic that takes down the life expectancy of African countries and all the rest of the world moves up into the corner where  we have long lives and small family and we have a completely new world .\', \' Family planning started in Vietnam and they went for smaller families and the United States up there is getting for longer life, keeping family size . In the eighties now, they give up communist planning and they go for market economy and  it moves faster even than social life . Today we have in Vietnam the same life expectancy and the same family size here in Vietnam, 19, 2003 as in United States, 1974 .\', " The richest 20% of the world\'s annual income is taken out of 74% and the poorest 20% take about 2% . This shows that the concept developing countries is extremely doubtful . In the middle, we have most of world population and they have now 24% of income .", " The World Bank\'s best projection from the World Bank is that this will happen. And we will not have a divided world. We have most people in the middle.  Of course, it\'s a logarithmic scale here.  But our concept of economy is growth with percent.  We look upon it as a possibility of percent increase. 0?  Then there was more of a hump.  And this is the pattern we have today.", \' In some countries, 99.7% of children survive to five years of age, others only 70% . OECD, Latin America, East Europe, East Asia, Arab states, South Asia and sub-Saharan Africa . The linearity is very strong between child survival and money.  But let me split sub.Saharan Africa.  Health is there and better health is up there. Sierra Leone down there, Mauritius up there .\', " There\'s a huge difference between Africa and Ghana is here in the middle . In Sierra Leone, humanitarian aid, here in Uganda, development aid,  time to invest,  there you can go for holiday . The differences are much bigger than the weakness of the data .", " Today we don\'t have to go to Cuba to find a healthy country in Latin America . Chile will have a lower child mortality than Cuba within some few years from now . And here we have high income countries in OECD and we get the whole pattern here of  the world which is more or less like this .", " South Korea is making a very, very fast advancement whereas Brazil is much slower . The countries are moving more or less in the same rate as money and health . But it seems you can move much faster if you are healthy first than if you\'re wealthy first . And to show that you can put on the way of United Arab Emirates, they came from here, a mineral country .", " If I split Uganda, there\'s quite a difference within Uganda . The richest 20% of Ugandans are there.  The poorest are down there. And if I go down and look at Nigeria, where there was such a terrible famine lastly, it\'s  like this. The 20% poorest of Nigeria is out here and the 20% richest of South Africa is there .", \' The improvement of the world must be highly contextualized, says Dr. Andrew Keen . Keen: "We have data in the United Nations, in the national statistical agencies, and in universities  and other non-governmental organizations" Keen: We find that students get very excited when they can use this.  And even more policymakers and the corporate sectors would like to see how the world is changing .\', " Linking Data to Design is a non-profit venture which aims to link data to design . It\'s not a new database you need, but we have wonderful design tools and more and more are added up here . What we really need is a search function where we can copy the data up to a searchable format and get it out in the world .", " Publicly funded data is down here and we would like flowers to grow out on the net . One of the crucial points is to make them searchable and then people can use the different design tool to animate it there . The present new head of UN statistic, he doesn\'t say it\'s impossible, only says we can\'t do it .", " The number of internet users are going up like this. s not so equal any longer.  And it\'s appearing here overlooking the United States almost like a ghost, isn\'t it?  It\'s pretty scary.  But I think it\'s very important to have all this information.  We need really to see it.  Instead of looking at this, I would like to end up by showing the internet users per  1000 .", " It\'s as if the world is flattening off, isn\'t it?  These countries are lifting more than the economy . But it\'s a nice tendency.  It will be very interesting to follow this over the year as I would like you to  be able to do with all the publicly funded data ."]'



### Example Of Shorter Summarized Text
<a id='short'></a>


```python
chunk_size = 1500  # change this to your desired chunk size
```


```python
# define the maximum length for summaries
max_summary_length = 100  # adjust this value as needed
```


```python
# changing the chunk size and max length to decrease thge summary output

# create a list to store the summarized text
summarized_text = []

for i in range(0, len(result), chunk_size):
    start = i
    end = i + chunk_size
    input_text = result[start:end]

    print("Input text:\n" + input_text)

    # generate a shorter summary by limiting the max_length
    out = summarizer(input_text, max_length=max_summary_length, min_length=10)  # adjust min_length as needed
    out = out[0]
    out = out['summary_text']

    print("Summarized text:\n" + out)

    summarized_text.append(out)
```
*Example**

    Input text:
     About ten years ago, I took on the task to teach global development to Swedish undergraduate  students.  That was after having spent about 20 years together with African institutions studying  hunger in Africa.  So I was sort of expected to know a little about the world.  And I started in our medical university, Karolinska Institute, an undergraduate course called  Global Health.  But when you get that opportunity, you get a little nervous.  I thought, these students coming to us actually have the highest grade you can get in Swedish  college system.  So I thought maybe they know everything I'm going to teach them about.  So I did a pre-test when they came.  And one of the questions from which I learned a lot was this one.  Which country has the highest child mortality of these five pairs?  And I put them together so that in each pair of country, one has twice the child mortality  of the other.  And this means that it's much bigger the difference than the uncertainty of the data.  I won't put you at a test here, but it's Turkey, which is highest there, Poland, Russia, Pakistan,  and South Africa.  And these were the results of the Swedish students.  I did it so I got a confidence interval, which was pretty narrow.  And I got happy, of course.  At one point, eight right answers out of five possible.  That means that there was a place for a professor of international health and for my course.  But one late night when I was compiling the report, I really realized my discovery. 
    Summarized text:
     Professor of international health at Karolinska Institute teaches global development to Swedish students . He says one of the questions from which he learned a lot was which country has the highest child mortality of these five pairs . Turkey, Poland, Russia, Pakistan, and South Africa have the highest rates of child mortality .
   

```python
 summarized_text_shorter = summarized_text
```


```python
# print length of summarized text
len(str(summarized_text_shorter))
```




    3437




```python
# print summarized text
str(summarized_text_shorter)
```



*
    '[\' Professor of international health at Karolinska Institute teaches global development to Swedish students . He says one of the questions from which he learned a lot was which country has the highest child mortality of these five pairs . Turkey, Poland, Russia, Pakistan, and South Africa have the highest rates of child mortality .\', \' I have shown that Swedish top students know statistically less about the world than chimpanzees . The problem for me was not ignorance.  It was preconceived ideas.  I did also an unethical study of the professors of the Karolinska Institute that hands out the Nobel Prize in medicine, and they are on par with the chimpanzee there.\', \' China is moving against better health, improving their, all the green Latin American countries are moving towards smaller families . Life expectancy at birth at birth from 30 years in some countries up to about 70 years .\', \' The data during the war indicate that even with all the death, there was an improvement  of life expectancy . By the end of the year, the family planning started in Vietnam and they went for smaller  families and the United States up there is getting for longer life, keeping family size . Today we have in Vietnam the same life expectancy and the same family size here in Vietnam, 19, 2003 as in United States, 1974 .\', " The richest 20% of world\'s annual income is taken out of 74% and the poorest 20% take out 74% . This shows that the concept developing countries is extremely doubtful . The problem in the world was the poverty in Asia, writes Andrew Hammond .", \' In some countries, 99.7% of children survive to five years of age, others only 70. e in the middle . There is a gap between OECD, Latin America, East Europe, East Asia,  Arab states, South Asia and sub-Saharan Africa .\', " There\'s a huge difference between Africa and Ghana is here in the middle . In Sierra Leone, humanitarian aid, here in Uganda, development aid,  time to invest,  there you can go for holiday . Data is often better than you think, but differences are bigger than the weakness of the data .", \' South Korea is making a very, very fast advancement whereas Brazil is much slower . The countries are moving more or less in the same rate as money and health . But it seems you can move much faster if you are healthy first .\', " A much more mainstream appearance of the world where all countries tend to use  their money better than they used in the past . Uganda, that today is where South Korea was  1960.  If I split Uganda, there\'s quite a difference within Uganda. The richest 20% of Ugandans are there.  The poorest are down there .", \' Linking Data to Design is a non-profit venture which aims to link data to design . The data is hidden down in the databases and the internet is there, but we have still not used it effectively . We are liberating UN data.  and in universities and other non-governmental organizations .\', " New UN statistic chief says data is not free to students or entrepreneurs . But he says it\'s important to have all this information available to the public . Data will be able to look at income distributions in completely new ways .", " iable you would like to have. The number of internet users are going up like this.  This is the GDP per capita.  But amazingly how well it fits to the economy of the countries.  That\'s why the $100 computer will be so important.  It\'s as if the world is flattening off, isn\'t it?"]'



### Conclusion
<a id='conclusion'></a>

This machine learning project demonstrates the power of combining various Python libraries, including Hugging Face Transformers and Whisper ASR, to automate the summarization of YouTube video transcripts. By following the outlined steps, one can easily extract key information from videos, making it a valuable tool for educational and research purposes.

The flexibility of this approach allows for fine-tuning the summarization process. By changing the chunk size, one can control how much text is processed at once, which can affect the granularity of the summaries. A larger chunk size may result in shorter summaries, as the model has more context to work with, while a smaller chunk size may yield slightly longer summaries for each segment.

Additionally, the max length parameter plays a crucial role in determining the length of the summarization output. Adjusting this parameter can result in either shorter or longer summaries. It's essential to strike a balance between brevity and informativeness to meet specific requirements.

In summary, this project not only showcases the automation of transcript summarization but also highlights the importance of parameter tuning in tailoring the summarization process to your needs, ultimately enhancing its utility for educational and research endeavors
