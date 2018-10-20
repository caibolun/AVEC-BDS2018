AUDIO_PATH='./recordings_audio' #the path of Bipolar Disorder Copurs audios in local server
TURKISH_TEXT_PATH='/Users/apple/AVEC2018/turkish_text/turkish_text_json' #the saving path of turkish text
GOOGLE_STORAGE= 'gs://ndhoho/recordings_audio/' #the file path of Bipolar Disorder Copurs audios in google strage

for i in $(ls $AUDIO_PATH)
do
echo `echo $i | cut -d \. -f 1`>>text1
done

for i in $(ls $TURKISH_TEXT_PATH)
do
echo `echo $i | cut -d \. -f 1`>>text2
done

# discard the processed targets
file_name_set=$(comm -23 text1 text2)
rm text1 text2
# file_name_set=$(cat text1) #the first time to run this code

for file_name in $file_name_set
do
    gcloud ml speech recognize-long-running \
    $GOOGLE_STORAGE$file_name'.wav' \
    --language-code='tr-TR'  \
    --include-word-time-offsets >> $TURKISH_TEXT_PATH/${file_name}.json

    wait
    
    # log
    echo $file_name is being processed
done

