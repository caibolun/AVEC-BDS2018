AUDIO_PATH='./recordings_audio' #the path of Bipolar Disorder Copurs audios in local server
TRANSLATION_PATH='./translation_origin'#the saving path of translation text
GOOGLE_STORAGE= 'gs://ndhoho/recordings_audio/' #the file path of Bipolar Disorder Copurs audios in google strage

for i in $(ls $AUDIO_PATH)
do
echo `echo $i | cut -d \. -f 1`>>text1
done


for i in $(ls $TRANSLATION_PATH)
do
echo `echo $i | cut -d \. -f 1`>>text2
done

# discard the processed targets
file_name_set=$(comm -23 text1 text2)
#file_name_set=$(cat text1) #the first time to run this code
rm text1 text2

for file_name in $file_name_set
do
    gcloud ml speech recognize-long-running \
    $GOOGLE_STORAGE$file_name'.wav' \
    --language-code='tr-TR'  \
    --include-word-time-offsets >> test.json

    #gcloud ml speech operations wait $(jq -r '.name' temp.json) >>test.json
    #wait
    #rm temp.json

    # extract the needed information from json file
    jq -r '.results[].alternatives[]' test.json >> test3.json
    rm test.json
    jq -r '[.words[0].startTime,.words[-1].endTime,.transcript,(.confidence|tostring)]|join(",")' test3.json >> record.csv
    rm test3.json

    # create new csv file, and add hearder to it
    echo 'start_time,end_time,transcript,confidence' >> $TRANSLATION_PATH/${file_name}.csv

    # log
    echo $file_name is being processed

    # process each line: translate each sentence
    cat record.csv | while read line
        do
        start_time=`echo $line | cut -d \, -f 1`
        end_time=`echo $line | cut -d \, -f 2`
        transcript=`echo $line | cut -d \, -f 3`
        transcript=${transcript//\'/ }
        confidence=`echo $line | cut -d \, -f 4`
        
        # use google cloud platform the translate transcripts
        curl -s -X POST -H "Content-Type: application/json" \
        -H "Authorization: Bearer "$(gcloud auth print-access-token) \
        --data "{
        'q': '$transcript',
        'source': 'tr',
        'target': 'en',
        'format': 'text'
        }" "https://translation.googleapis.com/language/translate/v2" >> translation.json
    
        wait
        transcript2=$( jq -r '.data.translations[].translatedText' translation.json)
        transcript2=${transcript2//\"/\"\"}
        combine=$start_time,$end_time,\"$transcript2\",$confidence
        rm translation.json 
        echo $combine >> $TRANSLATION_PATH/${file_name}.csv
    done
    rm record.csv
done
